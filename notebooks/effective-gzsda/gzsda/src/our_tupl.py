"""
m1 VAE + TUPL: train the domain-adversarial VAE, generate features, fit TUPL.

Policies (Xtc = real target seen, Xte = real target test):
- real_plus_src2tgt: Xsc/Xsu = cat(real source, source→target)
- real_plus_src2tgt_unseen: Xsc = real seen; Xsu = cat(real unseen, source→target)
- interp_src2tgt: Xsc/Xsu = α·real + (1-α)·source→target
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import numpy as np

from src.utils import (
    get_args,
    set_seed,
    get_datesets_and_loaders,
    get_trained_VAE_with_domain_classifier,
    prepare_report,
)
from src.tupl import TUPL

GENERATION_POLICIES = (
    "real_plus_src2tgt",
    "real_plus_src2tgt_unseen",
    "interp_src2tgt",
)

INTERP_ALPHA = 0.5

METHOD_PREFIX = "our_TUPL"

TensorDict = Dict[str, torch.Tensor]


def _to_float(x, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def _to_long(x, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)


def _cat_pair(x: torch.Tensor, x_gen: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([x, x_gen], dim=0), torch.cat([y, y], dim=0)


def _tupl_dict(
    Xsc: torch.Tensor,
    Ysc: torch.Tensor,
    Xsu: torch.Tensor,
    Ysu: torch.Tensor,
    Xtc: torch.Tensor,
    Ytc: torch.Tensor,
    Xte: torch.Tensor,
    Yte: torch.Tensor,
    seen_classes: torch.Tensor,
    unseen_classes: torch.Tensor,
) -> TensorDict:
    return dict(
        Xsc=Xsc.detach().clone(),
        Ysc=Ysc.detach().clone(),
        Xsu=Xsu.detach().clone(),
        Ysu=Ysu.detach().clone(),
        Xtc=Xtc.detach().clone(),
        Ytc=Ytc.detach().clone(),
        Xte=Xte.detach().clone(),
        Yte=Yte.detach().clone(),
        seen_classes=seen_classes.detach().clone(),
        unseen_classes=unseen_classes.detach().clone(),
    )


def extract_raw_splits(train_ds, test_ds, device) -> TensorDict:
    """Build TUPL-style splits from BaseTwoModalDataset train/test objects."""
    label_A = np.asarray(train_ds.label_A).squeeze().astype(np.int64).ravel()
    unseen_indicator = np.asarray(train_ds.unseenClass_B).squeeze().astype(np.int64).ravel()
    all_classes = np.unique(label_A)
    if len(unseen_indicator) != len(all_classes):
        raise ValueError(
            f"Unseen-class indicator length ({len(unseen_indicator)}) does not "
            f"match number of classes ({len(all_classes)})."
        )
    seen_np = all_classes[unseen_indicator == 0]
    unseen_np = all_classes[unseen_indicator == 1]
    is_unseen_A = np.isin(label_A, unseen_np)

    feature_A = np.asarray(train_ds.feature_A, dtype=np.float32)
    feature_B_train = np.asarray(train_ds.feature_B, dtype=np.float32)
    label_B_train = np.asarray(train_ds.label_B).squeeze().astype(np.int64).ravel()
    feature_B_test = np.asarray(test_ds.feature_B, dtype=np.float32)
    label_B_test = np.asarray(test_ds.label_B).squeeze().astype(np.int64).ravel()

    Xs = _to_float(feature_A, device)
    Ys = _to_long(label_A, device)
    mask = torch.as_tensor(is_unseen_A, dtype=torch.bool, device=device)

    return dict(
        Xs=Xs,
        Ys=Ys,
        source_unseen_mask=mask,
        Xsc=Xs[~mask],
        Ysc=Ys[~mask],
        Xsu=Xs[mask],
        Ysu=Ys[mask],
        Xtc=_to_float(feature_B_train, device),
        Ytc=_to_long(label_B_train, device),
        Xte=_to_float(feature_B_test, device),
        Yte=_to_long(label_B_test, device),
        seen_classes=_to_long(seen_np, device),
        unseen_classes=_to_long(unseen_np, device),
    )


@torch.no_grad()
def vae_forward_batched(
    vae,
    x: torch.Tensor,
    y: torch.Tensor,
    domain_id: int,
    device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (same-domain recon, cross-domain recon, encoder means)."""
    vae.eval()
    recon_same, recon_cross, means = [], [], []
    n = x.shape[0]
    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        d = torch.full((xb.shape[0],), int(domain_id), device=device, dtype=yb.dtype)
        recon, recon2, mean, _log_var, _z = vae(xb, yb, d=d)
        recon_same.append(recon)
        recon_cross.append(recon2)
        means.append(mean)
    return torch.cat(recon_same, dim=0), torch.cat(recon_cross, dim=0), torch.cat(means, dim=0)


@torch.no_grad()
def generate_vae_views(
    vae,
    raw: TensorDict,
    device,
    batch_size: int = 64,
) -> TensorDict:
    _src_same, src2tgt, _meanS = vae_forward_batched(
        vae, raw["Xs"], raw["Ys"], domain_id=0, device=device, batch_size=batch_size
    )
    mask = raw["source_unseen_mask"]
    return dict(
        src2tgt_seen=src2tgt[~mask],
        src2tgt_unseen=src2tgt[mask],
    )


def build_tupl_data(policy: str, raw: TensorDict, views: TensorDict) -> TensorDict:
    seen = raw["seen_classes"]
    unseen = raw["unseen_classes"]
    Ysc, Ysu, Ytc, Yte = raw["Ysc"], raw["Ysu"], raw["Ytc"], raw["Yte"]

    if policy == "real_plus_src2tgt":
        Xsc, Ysc_out = _cat_pair(raw["Xsc"], views["src2tgt_seen"], Ysc)
        Xsu, Ysu_out = _cat_pair(raw["Xsu"], views["src2tgt_unseen"], Ysu)
    elif policy == "real_plus_src2tgt_unseen":
        Xsc, Ysc_out = raw["Xsc"], Ysc
        Xsu, Ysu_out = _cat_pair(raw["Xsu"], views["src2tgt_unseen"], Ysu)
    elif policy == "interp_src2tgt":
        Xsc = INTERP_ALPHA * raw["Xsc"] + (1.0 - INTERP_ALPHA) * views["src2tgt_seen"]
        Xsu = INTERP_ALPHA * raw["Xsu"] + (1.0 - INTERP_ALPHA) * views["src2tgt_unseen"]
        Ysc_out, Ysu_out = Ysc, Ysu
    else:
        raise ValueError(
            f"Unknown generation policy {policy!r}; supported: {GENERATION_POLICIES}"
        )

    return _tupl_dict(
        Xsc, Ysc_out, Xsu, Ysu_out,
        raw["Xtc"], Ytc,
        raw["Xte"], Yte,
        seen, unseen,
    )


def _fit_evaluate_tupl(
    data: TensorDict,
    device,
    p: int = 1024,
    alpha: float = 1.0,
    m: int = 512,
    stage2_epochs: int = 2000,
    stage2_lr: float = 1e-4,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    feat_dim = int(data["Xsc"].shape[1])
    p_eff = min(p, feat_dim)
    model = TUPL(
        p=p_eff,
        alpha=alpha,
        m=m,
        stage2_epochs=stage2_epochs,
        stage2_lr=stage2_lr,
        device=device,
    )
    model.fit(data, verbose=verbose)
    return model.evaluate(data["Xte"], data["Yte"])


def _prepare_episode(
    args,
    DOMAIN_SET,
    DATA_DIR,
    DATASET_DETAILS,
    device,
):
    datasets, data_loaders = get_datesets_and_loaders(
        args=args,
        DOMAIN_SET=DOMAIN_SET,
        DATA_DIR=DATA_DIR,
        DATASET_DETAILS=DATASET_DETAILS,
    )
    vae = get_trained_VAE_with_domain_classifier(
        data_loaders=data_loaders,
        args=args,
        device=device,
    )
    raw = extract_raw_splits(datasets["train"], datasets["test"], device)
    batch_size = getattr(args, "batch_size", 64)
    views = generate_vae_views(vae, raw, device, batch_size=batch_size)
    return raw, views


def run_m1_tupl_policies(
    args,
    DOMAIN_SET,
    DATA_DIR,
    DATASET_DETAILS,
    policies: Sequence[str] = GENERATION_POLICIES,
    device=None,
    quiet: bool = False,
    **tupl_kwargs,
) -> Dict[str, Tuple[float, float, float]]:
    """Train the m1 VAE once, then fit TUPL for each generation policy.

    Returns percent (Accs, Accu, H) per policy, matching ``TUPL.evaluate``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, views = _prepare_episode(args, DOMAIN_SET, DATA_DIR, DATASET_DETAILS, device)

    metrics: Dict[str, Tuple[float, float, float]] = {}
    for policy in policies:
        data = build_tupl_data(policy, raw, views)
        acc_s, acc_u, h = _fit_evaluate_tupl(data, device=device, **tupl_kwargs)
        metrics[policy] = (acc_s, acc_u, h)
        if not quiet:
            print(
                "[{}] seen acc:{:2.4f}, unseen acc:{:2.4f}, H:{:2.4f}".format(
                    policy, acc_s / 100.0, acc_u / 100.0, h / 100.0
                )
            )
    return metrics


def run_m1_tupl(
    args,
    DOMAIN_SET,
    DATA_DIR,
    DATASET_DETAILS,
    policy: str = "real_plus_src2tgt",
    device=None,
    quiet: bool = False,
    **tupl_kwargs,
) -> Tuple[float, float, float]:
    """Train m1 VAE then TUPL for a single generation policy. Metrics in percent."""
    metrics = run_m1_tupl_policies(
        args,
        DOMAIN_SET,
        DATA_DIR,
        DATASET_DETAILS,
        policies=(policy,),
        device=device,
        quiet=quiet,
        **tupl_kwargs,
    )
    return metrics[policy]


def run_all_senario_m1_tupl(
    DOMAIN_SET,
    DATA_DIR,
    DATASET_DETAILS,
    policies: Sequence[str] = GENERATION_POLICIES,
    input_dim: int = 2048,
    num_trial: int = 5,
    method_prefix: str = METHOD_PREFIX,
    device=None,
    quiet: bool = False,
    pairs=None,
    **tupl_kwargs,
) -> Dict[str, Dict[str, str]]:
    """All domain pairs × trials; VAE trained once per episode, all policies evaluated.

    Returns ``{our_TUPL_<policy>: {scenario: prepare_report string}}``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial_rows: Dict[str, Dict[str, List[Tuple]]] = {p: {} for p in policies}

    if pairs is None:
        pairs = [
            (s, t)
            for s in range(len(DOMAIN_SET))
            for t in range(len(DOMAIN_SET))
            if s != t
        ]
    else:
        pairs = list(pairs)

    for s, t in pairs:
        scenario = "%s -> %s" % (DOMAIN_SET[s], DOMAIN_SET[t])
        print(scenario)
        for policy in policies:
            trial_rows[policy][scenario] = []

        for i in range(num_trial):
            args = get_args(
                trialIndex=i,
                sourceDomainIndex=s,
                targetDomainIndex=t,
                input_dim=input_dim,
            )
            set_seed(args)
            metrics = run_m1_tupl_policies(
                args,
                DOMAIN_SET,
                DATA_DIR,
                DATASET_DETAILS,
                policies=policies,
                device=device,
                quiet=quiet,
                **tupl_kwargs,
            )
            for policy, (acc_s, acc_u, _h) in metrics.items():
                trial_rows[policy][scenario].append(
                    (None, None, acc_s / 100.0, acc_u / 100.0)
                )

        for policy in policies:
            report = prepare_report(trial_rows[policy][scenario])
            print(f"{method_prefix}_{policy}")
            print(report)

    result: Dict[str, Dict[str, str]] = {}
    for policy in policies:
        result[f"{method_prefix}_{policy}"] = {
            scenario: prepare_report(rows)
            for scenario, rows in trial_rows[policy].items()
        }
    return result
