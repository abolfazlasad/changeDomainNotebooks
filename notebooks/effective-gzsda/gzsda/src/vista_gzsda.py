"""Feature-space VisTA for GZSDA (ActionStyle / MotionCLIP embeddings).

Official VisTA (CI-UDA) encodes RGB images with CLIP ViT and uses Grad-CAM
visual-attention consistency. ActionStyle has no images — only 512-d MotionCLIP
embeddings already in CLIP ViT-B/32 space — so this module:

- skips the image encoder
- replaces Grad-CAM VAC with cosine retrieval from the other domain's
  attribute dictionary (VisTA paper ablation "w/o VAC")
- uses GZSDA splits: labeled source (all classes), unlabeled target seen-train,
  test on target seen + unseen with Acc_s / Acc_u / H-mean

Does not import Dassl. CLIP is used as a frozen text encoder only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.tupl import TUPLDataset, macro_per_class_accuracy

ACTIONSTYLE_CLASS_NAMES = ["punch", "jump", "kick", "walk"]

_CLIP_CACHE: Dict[str, nn.Module] = {}


@dataclass
class VistaConfig:
    backbone: str = "ViT-B/32"
    n_attr: int = 4
    n_select: int = 2
    n_ctx: int = 4
    batch_size: int = 8
    epochs: int = 15
    lr: float = 3e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    gamma: float = 0.6
    tau: float = 0.5
    lam_1: float = 1.0
    lam_2: float = 1.0
    lam_3: float = 1.0
    debias: bool = True
    qhat_momentum: float = 0.99
    seed: int = 1


def load_clip(device: torch.device, backbone: str = "ViT-B/32") -> nn.Module:
    """Load OpenAI CLIP once per process/device (text encoder + token embedding)."""
    key = f"{backbone}:{device}"
    if key not in _CLIP_CACHE:
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "OpenAI CLIP is required for feature-space VisTA. Install with:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            ) from exc
        model, _ = clip.load(backbone, device=device, jit=False)
        model = model.float().eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _CLIP_CACHE[key] = model
    return _CLIP_CACHE[key]


def jsd(out1: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two logit tensors."""
    prob1 = F.softmax(out1, dim=1)
    prob2 = F.softmax(out2, dim=1)
    mix = 0.5 * (prob1 + prob2)
    kl_pm = F.kl_div(F.log_softmax(out1, dim=1), mix, reduction="batchmean")
    kl_qm = F.kl_div(F.log_softmax(out2, dim=1), mix, reduction="batchmean")
    return 0.5 * (kl_pm + kl_qm)


def debias_probs(logits: torch.Tensor, qhat: torch.Tensor, tau: float) -> torch.Tensor:
    return F.softmax(logits - tau * torch.log(qhat.clamp_min(1e-6)), dim=1)


def update_qhat(
    probs: torch.Tensor,
    qhat: torch.Tensor,
    momentum: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is not None and mask.sum() > 0:
        weights = mask.unsqueeze(-1)
        mean_prob = (probs.detach() * weights).sum(0, keepdim=True) / weights.sum()
    else:
        mean_prob = probs.detach().mean(0, keepdim=True)
    return momentum * qhat + (1.0 - momentum) * mean_prob


def kmeans_centers(features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n = features.shape[0]
    k = max(1, min(n_clusters, n))
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    km.fit(features)
    centers = km.cluster_centers_.astype(np.float32)
    if k < n_clusters:
        pad = np.random.RandomState(seed).randn(n_clusters - k, features.shape[1]).astype(
            np.float32
        )
        pad /= np.linalg.norm(pad, axis=1, keepdims=True).clip(min=1e-6)
        centers = np.concatenate([centers, pad], axis=0)
    norms = np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-6)
    return centers / norms


class FrozenTextEncoder:
    """CLIP text tower without registering it as an nn.Module (keeps the cache intact)."""

    def __init__(self, clip_model: nn.Module):
        self.clip_model = clip_model

    def __call__(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        clip_model = self.clip_model
        x = prompts + clip_model.positional_embedding.type(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(prompts.dtype)
        eos = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), eos] @ clip_model.text_projection


class PromptLearner(nn.Module):
    def __init__(
        self,
        class_names: Sequence[str],
        clip_model: nn.Module,
        text_attr: nn.Parameter,
        n_ctx: int,
        n_select: int,
        n_attr: int,
        target_domain: str,
    ):
        super().__init__()
        import clip

        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.n_cls = len(class_names)
        self.attr_L = n_select
        self.attr_N = n_attr
        self.attr_M = n_ctx
        self.text_attr = text_attr
        names = [name.replace("_", " ") for name in class_names]

        prompt_prefix = " ".join(["x"] * (n_ctx * n_select))
        prompts = [f"{prompt_prefix} {name}." for name in names]
        style = target_domain.replace("_", " ")
        naive_prompts = [f"a {style} motion of a person {name}." for name in names]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized = torch.cat([clip.tokenize(p) for p in naive_prompts])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(next(clip_model.parameters()).device))
            embedding = embedding.type(clip_model.dtype)
            fixed = clip_model.encode_text(naive_tokenized.to(next(clip_model.parameters()).device))
            fixed = fixed.type(clip_model.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + (n_ctx * n_select) :, :])
        self.register_buffer("fixed_embeddings", fixed)

        nc_prompts = [prompt_prefix + "."]
        nc_tokenized = torch.cat([clip.tokenize(p) for p in nc_prompts])
        self.register_buffer("nc_tokenized_prompts", nc_tokenized)
        with torch.no_grad():
            nc_emb = clip_model.token_embedding(nc_tokenized.to(next(clip_model.parameters()).device))
            nc_emb = nc_emb.type(clip_model.dtype)
        self.register_buffer("nc_token_prefix", nc_emb[:, :1, :])
        self.register_buffer("nc_token_suffix", nc_emb[:, 1 + n_ctx :, :])

    def compose(self, indices: torch.Tensor, target_dict: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build class prompts from selected attribute indices. indices: [B, L]."""
        batch = indices.shape[0]
        offset = self.attr_N if target_dict else 0
        selected = self.text_attr[indices + offset]
        ctx = selected.view(batch, self.attr_M * self.attr_L, self.ctx_dim)
        prefix = self.token_prefix.unsqueeze(0).expand(batch, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(batch, -1, -1, -1)
        ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        prompts = torch.cat([prefix, ctx, suffix], dim=2)
        prompts = prompts.reshape(batch * self.n_cls, -1, self.ctx_dim)
        tokenized = self.tokenized_prompts.unsqueeze(0).expand(batch, -1, -1)
        tokenized = tokenized.reshape(batch * self.n_cls, -1)
        return prompts, tokenized

    def only_prefix(self, target: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = self.text_attr[self.attr_N :] if target else self.text_attr[: self.attr_N]
        prompt_size = ctx.shape[0]
        nc_tokenized = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        return torch.cat([prefix, ctx, suffix], dim=1), nc_tokenized


class FeatureVisTA(nn.Module):
    """VisTA attribute model that takes CLIP-space embeddings instead of images."""

    def __init__(
        self,
        clip_model: nn.Module,
        class_names: Sequence[str],
        target_domain: str,
        cfg: VistaConfig,
        vis_attr: torch.Tensor,
    ):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_class = len(class_names)
        self.attr_L = cfg.n_select
        self.num_prompt = cfg.n_attr
        self.text_encoder = FrozenTextEncoder(clip_model)
        text_attr = nn.Parameter(
            torch.empty(2 * cfg.n_attr, cfg.n_ctx, ctx_dim, dtype=torch.float32)
        )
        nn.init.normal_(text_attr, std=0.02)
        self.prompt_learner = PromptLearner(
            class_names=class_names,
            clip_model=clip_model,
            text_attr=text_attr,
            n_ctx=cfg.n_ctx,
            n_select=cfg.n_select,
            n_attr=cfg.n_attr,
            target_domain=target_domain,
        )
        self.register_buffer("vis_attr", vis_attr.float())
        self.register_buffer(
            "logit_scale", clip_model.logit_scale.detach().float().clone()
        )

    def _topk(self, features: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        k = min(self.attr_L, keys.shape[0])
        _, indices = (features @ keys.t()).topk(k=k, dim=1, largest=True)
        return indices

    def _logits_from_indices(
        self, features: torch.Tensor, indices: torch.Tensor, target_dict: bool
    ) -> torch.Tensor:
        prompts, tokenized = self.prompt_learner.compose(indices, target_dict=target_dict)
        text_features = self.text_encoder(prompts, tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(features.shape[0], self.n_class, -1)
        scale = self.logit_scale.exp()
        return scale * (features.unsqueeze(1) * text_features).sum(-1)

    def diversity_and_hp(
        self, text_features_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nc_s, tok_s = self.prompt_learner.only_prefix(target=False)
        nc_t, tok_t = self.prompt_learner.only_prefix(target=True)
        feat_s = self.text_encoder(nc_s, tok_s)
        feat_t = self.text_encoder(nc_t, tok_t)
        feat_s = feat_s / feat_s.norm(dim=-1, keepdim=True)
        feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
        eye = torch.eye(self.num_prompt, dtype=torch.bool, device=feat_s.device)
        dis_s = feat_s @ feat_s.t()
        dis_t = feat_t @ feat_t.t()
        loss_div = dis_s[~eye].abs().mean() + dis_t[~eye].abs().mean()
        fixed = self.prompt_learner.fixed_embeddings
        fixed = fixed / fixed.norm(dim=-1, keepdim=True)
        loss_hp = F.l1_loss(text_features_s, fixed.unsqueeze(0).expand_as(text_features_s))
        return loss_div, loss_hp

    def forward(
        self,
        feat_s: Optional[torch.Tensor] = None,
        feat_t: Optional[torch.Tensor] = None,
        test: bool = False,
    ):
        vis_s = self.vis_attr[: self.num_prompt]
        vis_t = self.vis_attr[self.num_prompt :]

        if test:
            assert feat_t is not None
            feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)
            indices_tt = self._topk(feat_t, vis_t)
            return self._logits_from_indices(feat_t, indices_tt, target_dict=True)

        assert feat_s is not None and feat_t is not None
        feat_s = feat_s / feat_s.norm(dim=-1, keepdim=True)
        feat_t = feat_t / feat_t.norm(dim=-1, keepdim=True)

        # Same-domain retrieval + cosine cross-domain retrieval (no Grad-CAM VAC).
        indices_ss = self._topk(feat_s, vis_s)
        indices_tt = self._topk(feat_t, vis_t)
        indices_st = self._topk(feat_s, vis_t)
        indices_ts = self._topk(feat_t, vis_s)

        prompts_ss, tok_ss = self.prompt_learner.compose(indices_ss, target_dict=False)
        text_ss = self.text_encoder(prompts_ss, tok_ss)
        text_ss = text_ss / text_ss.norm(dim=-1, keepdim=True)
        text_ss = text_ss.view(feat_s.shape[0], self.n_class, -1)

        logits_ss = self.logit_scale.exp() * (feat_s.unsqueeze(1) * text_ss).sum(-1)
        logits_st = self._logits_from_indices(feat_s, indices_st, target_dict=True)
        logits_ts = self._logits_from_indices(feat_t, indices_ts, target_dict=False)
        logits_tt = self._logits_from_indices(feat_t, indices_tt, target_dict=True)
        loss_div, loss_hp = self.diversity_and_hp(text_ss)
        return logits_ss, logits_st, logits_ts, logits_tt, loss_div, loss_hp


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _build_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(features.float(), labels.long())
    effective = min(batch_size, max(1, len(ds)))
    return DataLoader(ds, batch_size=effective, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def evaluate_vista(
    model: FeatureVisTA,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    seen_classes: torch.Tensor,
    unseen_classes: torch.Tensor,
    n_class: int,
    batch_size: int = 64,
) -> Tuple[np.ndarray, float, float, float]:
    model.eval()
    preds = []
    for start in range(0, x_te.shape[0], batch_size):
        chunk = x_te[start : start + batch_size]
        logits = model(feat_t=chunk, test=True)
        preds.append(logits.argmax(dim=-1))
    y_pred = torch.cat(preds, dim=0)
    acc_s, acc_u, _h = macro_per_class_accuracy(y_te, y_pred, seen_classes, unseen_classes)
    acc_per_class = np.zeros((n_class,), dtype=np.float64)
    y_true_np = y_te.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    for c in range(n_class):
        mask = y_true_np == c
        if mask.any():
            acc_per_class[c] = (y_pred_np[mask] == y_true_np[mask]).mean()
    overall = float(np.mean(acc_per_class)) if n_class else 0.0
    return acc_per_class, overall, acc_s / 100.0, acc_u / 100.0


def train_feature_vista(
    data: Dict[str, torch.Tensor],
    class_names: Sequence[str],
    target_domain: str,
    device: torch.device,
    cfg: VistaConfig,
    quiet: bool = True,
) -> FeatureVisTA:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    x_s = torch.cat([data["Xsc"], data["Xsu"]], dim=0)
    y_s = torch.cat([data["Ysc"], data["Ysu"]], dim=0)
    x_t = data["Xtc"]
    if x_t.shape[0] == 0:
        raise ValueError("Target seen-train split is empty; cannot run VisTA.")

    vis_s = kmeans_centers(x_s.detach().cpu().numpy(), cfg.n_attr, cfg.seed)
    vis_t = kmeans_centers(x_t.detach().cpu().numpy(), cfg.n_attr, cfg.seed + 1)
    vis_attr = torch.from_numpy(np.concatenate([vis_s, vis_t], axis=0)).to(device)

    clip_model = load_clip(device, backbone=cfg.backbone)
    model = FeatureVisTA(
        clip_model=clip_model,
        class_names=class_names,
        target_domain=target_domain,
        cfg=cfg,
        vis_attr=vis_attr,
    ).to(device)
    model.train()

    params = [p for p in model.prompt_learner.parameters() if p.requires_grad]
    optim = torch.optim.SGD(
        params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(cfg.epochs, 1))

    src_loader = _build_loader(x_s, y_s, cfg.batch_size, shuffle=True)
    tgt_loader = _build_loader(x_t, torch.zeros(x_t.shape[0], dtype=torch.long), cfg.batch_size, shuffle=True)
    n_batches = max(len(src_loader), len(tgt_loader))
    qhat = torch.ones(1, len(class_names), device=device) / len(class_names)

    epoch_bar = tqdm(
        range(cfg.epochs),
        desc="vista epochs",
        leave=False,
        unit="ep",
    )
    for epoch in epoch_bar:
        src_iter = _cycle(src_loader)
        tgt_iter = _cycle(tgt_loader)
        running = 0.0
        for _ in range(n_batches):
            feat_s, label_s = next(src_iter)
            feat_t, _ = next(tgt_iter)
            feat_s = feat_s.to(device)
            label_s = label_s.to(device)
            feat_t = feat_t.to(device)

            logits_ss, logits_st, logits_ts, logits_tt, loss_div, loss_hp = model(
                feat_s=feat_s, feat_t=feat_t, test=False
            )
            loss_s = F.cross_entropy(logits_ss, label_s)
            if cfg.debias:
                probs_tt = debias_probs(logits_tt, qhat, tau=cfg.tau)
            else:
                probs_tt = torch.softmax(logits_tt, dim=-1)
            max_probs, label_p = torch.max(probs_tt, dim=-1)
            mask_ge = max_probs.ge(cfg.gamma).float()
            if cfg.debias:
                qhat = update_qhat(
                    torch.softmax(logits_tt.detach(), dim=-1),
                    qhat,
                    momentum=cfg.qhat_momentum,
                    mask=mask_ge,
                )
            if mask_ge.sum() == 0:
                loss_t = logits_tt.new_zeros(())
            else:
                loss_t = (
                    F.cross_entropy(logits_tt, label_p, reduction="none") * mask_ge
                ).sum() / mask_ge.sum()
            loss_con = jsd(logits_tt, logits_ts) + jsd(logits_ss, logits_st)
            loss = (
                loss_s
                + loss_t
                + cfg.lam_1 * loss_con
                + cfg.lam_2 * loss_hp
                + cfg.lam_3 * loss_div
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
        sched.step()
        avg_loss = running / n_batches
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")
        if not quiet:
            print(
                f"  vista epoch {epoch + 1}/{cfg.epochs}  loss={avg_loss:.4f}"
            )
    return model


def run_vista(
    args,
    DOMAIN_SET: Sequence[str],
    DATA_DIR: str,
    DATASET_DETAILS: Dict,
    class_names: Optional[Sequence[str]] = None,
    device=None,
    quiet: bool = True,
    cfg: Optional[VistaConfig] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Train feature-space VisTA on one GZSDA episode.

    Returns ``(acc_per_class, overall, acc_seen, acc_unseen)`` as fractions so
    ``run_all_senario`` / ``prepare_report`` can be used unchanged.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg or VistaConfig()
    if hasattr(args, "seed") and args.seed is not None:
        cfg.seed = int(args.seed)
    names = list(class_names or ACTIONSTYLE_CLASS_NAMES)
    target_name = DOMAIN_SET[args.targetDomainIndex]

    ds = TUPLDataset(
        domain_set=list(DOMAIN_SET),
        data_dir=DATA_DIR,
        source_domain_index=args.sourceDomainIndex,
        target_domain_index=args.targetDomainIndex,
        trial_index=args.trialIndex,
        dataset_details=DATASET_DETAILS,
        device=str(device),
    )
    data = ds.as_tensors(device=device)
    n_class = int(torch.cat([data["seen_classes"], data["unseen_classes"]]).max().item()) + 1
    if len(names) < n_class:
        names = names + [f"class_{i}" for i in range(len(names), n_class)]
    elif len(names) > n_class:
        names = names[:n_class]

    model = train_feature_vista(
        data=data,
        class_names=names,
        target_domain=target_name,
        device=device,
        cfg=cfg,
        quiet=quiet,
    )
    acc_per_class, overall, acc_s, acc_u = evaluate_vista(
        model,
        data["Xte"],
        data["Yte"],
        data["seen_classes"],
        data["unseen_classes"],
        n_class=n_class,
    )
    if not quiet:
        h = 0.0 if (acc_s + acc_u) == 0 else 2 * acc_s * acc_u / (acc_s + acc_u)
        print(
            "seen acc:{:2.4f}, unseen acc:{:2.4f}, H:{:2.4f}".format(acc_s, acc_u, h)
        )
    return acc_per_class, overall, acc_s, acc_u


if __name__ == "__main__":
    # From notebooks/effective-gzsda/gzsda:  python -m src.vista_gzsda
    from src.utils import get_args

    DOMAIN_SET = [
        "angry", "childlike", "depressed", "neutral", "old", "proud", "strutting",
    ]
    DATA_DIR = "./data/ActionStyleDataset/"
    DATASET_DETAILS = {
        "prefix": "ActionStyle-",
        "suffix": "-clip.mat",
        "resnet_feature": "clip_features",
        "split_file_name": "instanceSplit_actionStyle_unseen2.mat",
    }
    args = get_args(trialIndex=0, sourceDomainIndex=0, targetDomainIndex=1, input_dim=512)
    acc_per_class, overall, acc_s, acc_u = run_vista(
        args,
        DOMAIN_SET,
        DATA_DIR,
        DATASET_DETAILS,
        quiet=False,
    )
    print("acc_per_class", acc_per_class)
    print("overall", overall)
