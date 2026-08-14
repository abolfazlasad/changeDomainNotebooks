"""
TUPL — single-file implementation.

Generalized Zero-Shot Domain Adaptation with Target Unseen Class Prototype
Learning (Li, Fang & Chen, Neural Computing and Applications, 2022).

Merged from: configs.py, datasets.py, utils.py, subspace.py, prototype_nets.py,
tupl.py, inspect_mat.py, train.py, test_synthetic.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

# Paper Table 6 targets (H only; other pairs not published in paper/tupl.md)
PAPER_OFFICE31_H_TARGETS = {
    "A -> D": 92.6,
    "A -> W": 92.8,
    "D -> A": 76.9,
}

MetricMode = Literal["macro", "sample"]
FusionMode = Literal["phi", "psi", "avg", "subspace"]
GraphKernelMode = Literal["paper", "median"]
LossReduction = Literal["sum", "mean"]

# =============================================================================
# Config presets (configs.py)
# =============================================================================

DATASET_CONFIGS = {
    "officehome": {
        "subdir": "OfficeHome/",
        "domain_set": ["Art", "Clipart", "Product", "RealWorld"],
        "backbones": {
            "resnet50": {
                "prefix": "OfficeHome-",
                "suffix": "-resnet50-noft.mat",
                "resnet_feature": "resnet50",
            },
            "vgg16": {
                "prefix": "OfficeHome-",
                "suffix": "-vgg16-noft.mat",
                "resnet_feature": "vgg16",
            },
        },
        "split_file_name": "instanceSplit_officehome_unseen30.mat",
    },
    "office31": {
        "subdir": "Office31/",
        "domain_set": ["A", "D", "W"],
        "backbones": {
            "resnet50": {
                "prefix": "office-",
                "suffix": "-resnet50-noft.mat",
                "resnet_feature": "resnet50_features",
                "num_seen": 16,
                "num_unseen": 15,
            },
        },
        "split_file_name": "instanceSplit_office31_unseen15.mat",
    },
    "xraybaggage20": {
        "subdir": "XrayBaggage20/",
        "domain_set": ["regu", "xray"],
        "backbones": {
            "resnet101": {
                "prefix": "XrayDataset-",
                "suffix": "-resnet101-noft.mat",
                "resnet_feature": "resnet101_features",
            },
        },
        "split_file_name": "instanceSplit_xrayDataset_unseen10.mat",
    },
    "actionstyle": {
        "subdir": "ActionStyleDataset/",
        "domain_set": [
            "angry", "childlike", "depressed", "neutral",
            "old", "proud", "strutting",
        ],
        "backbones": {
            "clip": {
                "prefix": "ActionStyle-",
                "suffix": "-clip.mat",
                "resnet_feature": "clip_features",
            },
        },
        "split_file_name": "instanceSplit_actionStyle_unseen2.mat",
    },
}

# Result file names aligned with refactored-*.ipynb / result/csv|json
DATASET_RESULT_NAMES = {
    "office31": "office31",
    "officehome": "officeHome",
    "xraybaggage20": "xray",
    "actionstyle": "actionStyle",
}


def get_default_backbone(dataset: str) -> str:
    return next(iter(DATASET_CONFIGS[dataset]["backbones"]))


def get_result_name(dataset: str) -> str:
    return DATASET_RESULT_NAMES.get(dataset, dataset)


def get_dataset_details(dataset: str, backbone: str, data_root: str = "data/"):
    cfg = DATASET_CONFIGS[dataset]
    if backbone not in cfg["backbones"]:
        raise ValueError(
            f"backbone {backbone!r} not available for {dataset!r}; "
            f"choose one of {list(cfg['backbones'])}"
        )
    dd = dict(cfg["backbones"][backbone])
    dd["split_file_name"] = cfg["split_file_name"]
    data_dir = data_root.rstrip("/") + "/" + cfg["subdir"]
    return cfg["domain_set"], data_dir, dd


def get_num_trials(dataset: str, data_root: str = "../data/", backbone: Optional[str] = None) -> int:
    """Number of random trials stored in the dataset split file."""
    backbone = backbone or next(iter(DATASET_CONFIGS[dataset]["backbones"]))
    _, data_dir, dd = get_dataset_details(dataset, backbone, data_root=data_root)
    split = scipy.io.loadmat(data_dir + dd["split_file_name"])
    return int(split["targetDomain_splitFlag"].shape[1])


def iter_domain_pairs(
    domain_set: Sequence[str], include_same_domain: bool = False
) -> Iterable[Tuple[int, int]]:
    n = len(domain_set)
    for source in range(n):
        for target in range(n):
            if include_same_domain or source != target:
                yield source, target


def set_seed(seed: int) -> None:
    """Set NumPy and PyTorch seeds for reproducible stage-2 optimization."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def frobenius_squared_loss(
    a: torch.Tensor, b: torch.Tensor, reduction: LossReduction = "sum"
) -> torch.Tensor:
    """Frobenius norm squared; paper uses ||·||_F^2 (sum of squared elements)."""
    diff2 = (a - b) ** 2
    return diff2.sum() if reduction == "sum" else diff2.mean()


# =============================================================================
# Classification utilities (utils.py)
# =============================================================================

@torch.no_grad()
def nearest_prototype_classify(
    X: torch.Tensor, prototypes: torch.Tensor, classes: torch.Tensor, batch_size: int = None
) -> torch.Tensor:
    if batch_size is None:
        d2 = torch.cdist(X, prototypes, p=2)
        idx = d2.argmin(dim=1)
        return classes[idx]

    preds = []
    for start in range(0, X.shape[0], batch_size):
        chunk = X[start : start + batch_size]
        d2 = torch.cdist(chunk, prototypes, p=2)
        idx = d2.argmin(dim=1)
        preds.append(classes[idx])
    return torch.cat(preds, dim=0)


def per_class_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    classes: torch.Tensor,
) -> torch.Tensor:
    """Per-class accuracy for each class in `classes` (0 if class absent in y_true)."""
    accs = []
    for c in classes:
        mask = y_true == c
        if mask.any():
            accs.append((y_pred[mask] == y_true[mask]).float().mean())
        else:
            accs.append(torch.zeros((), device=y_true.device))
    return torch.stack(accs)


def macro_per_class_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    seen_classes: torch.Tensor,
    unseen_classes: torch.Tensor,
) -> Tuple[float, float, float]:
    """Macro mean of per-class accuracies (matches GZSDA repo convention)."""
    acc_seen = per_class_accuracy(y_true, y_pred, seen_classes)
    acc_unseen = per_class_accuracy(y_true, y_pred, unseen_classes)
    acc_s = acc_seen.mean().item()
    acc_u = acc_unseen.mean().item()
    h = 0.0 if (acc_s + acc_u) == 0 else 2 * acc_s * acc_u / (acc_s + acc_u)
    return acc_s * 100, acc_u * 100, h * 100


def sample_level_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    seen_classes: torch.Tensor,
    unseen_classes: torch.Tensor,
) -> Tuple[float, float, float]:
    seen_mask = torch.isin(y_true, seen_classes)
    unseen_mask = torch.isin(y_true, unseen_classes)

    acc_s = (
        (y_pred[seen_mask] == y_true[seen_mask]).float().mean().item()
        if seen_mask.any()
        else 0.0
    )
    acc_u = (
        (y_pred[unseen_mask] == y_true[unseen_mask]).float().mean().item()
        if unseen_mask.any()
        else 0.0
    )

    h = 0.0 if (acc_s + acc_u) == 0 else 2 * acc_s * acc_u / (acc_s + acc_u)
    return acc_s * 100, acc_u * 100, h * 100


def harmonic_mean_accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    seen_classes: torch.Tensor,
    unseen_classes: torch.Tensor,
    metric: MetricMode = "macro",
) -> Tuple[float, float, float]:
    if metric == "macro":
        return macro_per_class_accuracy(y_true, y_pred, seen_classes, unseen_classes)
    return sample_level_accuracy(y_true, y_pred, seen_classes, unseen_classes)


# =============================================================================
# Stage 1 — common subspace (subspace.py)
# =============================================================================

def _pairwise_sq_dists(X: torch.Tensor) -> torch.Tensor:
    sq_norms = (X ** 2).sum(dim=1, keepdim=True)
    d2 = sq_norms + sq_norms.T - 2.0 * X @ X.T
    return d2.clamp_min(0.0)


def build_similarity_graph(
    X: torch.Tensor,
    y: torch.Tensor,
    sigma: Optional[float] = None,
    kernel_mode: GraphKernelMode = "paper",
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = X.shape[0]
    d2 = _pairwise_sq_dists(X)

    if kernel_mode == "paper":
        # Paper Eq. (2): exp(-||X_i - X_j||^2) with no bandwidth scaling.
        sigma2 = torch.as_tensor(1.0, device=X.device, dtype=X.dtype)
    elif sigma is None:
        mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
        sigma2 = d2[mask].median().clamp_min(1e-8)
    else:
        sigma2 = torch.as_tensor(sigma ** 2, device=X.device, dtype=X.dtype)

    same_class = y.view(-1, 1) == y.view(1, -1)
    Q = torch.where(same_class, torch.exp(-d2 / sigma2), torch.zeros_like(d2))
    Q.fill_diagonal_(0.0)

    D = torch.diag(Q.sum(dim=1))
    L = D - Q
    return Q, L


def class_prototypes(X: torch.Tensor, y: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    protos = []
    for c in classes:
        mask = y == c
        protos.append(X[mask].mean(dim=0))
    return torch.stack(protos, dim=0)


def one_hot_prototype_matrix(
    y: torch.Tensor, classes: torch.Tensor, protos: torch.Tensor
) -> torch.Tensor:
    class_to_idx = {int(c.item()): i for i, c in enumerate(classes)}
    idx = torch.tensor([class_to_idx[int(v.item())] for v in y], device=y.device)
    return protos[idx]


def centering_matrix(n: int, device, dtype) -> torch.Tensor:
    I = torch.eye(n, device=device, dtype=dtype)
    ones = torch.ones(n, n, device=device, dtype=dtype) / n
    return I - ones


def solve_generalized_eig_top(
    A: torch.Tensor, B: torch.Tensor, p: int, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = A.shape[0]
    orig_dtype = A.dtype
    A64 = A.double()
    B64 = B.double()

    reg_eps = eps
    for _ in range(12):
        try:
            A_reg = A64 + reg_eps * torch.eye(d, device=A.device, dtype=torch.float64)
            Lc = torch.linalg.cholesky(A_reg)
            break
        except RuntimeError:
            reg_eps *= 10.0
    else:
        raise RuntimeError(
            "Failed to factorize generalized eigenproblem matrix A after repeated regularization."
        )

    Lc_inv = torch.linalg.solve_triangular(
        Lc, torch.eye(d, device=A.device, dtype=torch.float64), upper=False
    )

    C = Lc_inv @ B64 @ Lc_inv.T
    C = 0.5 * (C + C.T)

    eigvals, eigvecs = torch.linalg.eigh(C)
    top = eigvecs[:, -p:].flip(dims=[1])
    W = Lc_inv.T @ top

    return W.to(orig_dtype), eigvals.flip(dims=[0])[:p].to(orig_dtype)


class SubspaceLearner:
    def __init__(
        self,
        p: int = 1024,
        alpha: float = 1.0,
        sigma: Optional[float] = None,
        eps: float = 1e-5,
        kernel_mode: GraphKernelMode = "paper",
    ):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
        self.eps = eps
        self.kernel_mode = kernel_mode
        self.W: Optional[torch.Tensor] = None

    def fit(
        self,
        Xsc: torch.Tensor,
        Ysc: torch.Tensor,
        Xsu: torch.Tensor,
        Ysu: torch.Tensor,
        Xtc: torch.Tensor,
        Ytc: torch.Tensor,
        seen_classes: torch.Tensor,
        unseen_classes: torch.Tensor,
    ) -> "SubspaceLearner":
        device, dtype = Xsc.device, Xsc.dtype

        X = torch.cat([Xsc, Xsu, Xtc], dim=0)
        y = torch.cat([Ysc, Ysu, Ytc], dim=0)
        n, d = X.shape

        _, L = build_similarity_graph(X, y, sigma=self.sigma, kernel_mode=self.kernel_mode)

        Ps = class_prototypes(Xsc, Ysc, seen_classes)
        Pu = class_prototypes(Xsu, Ysu, unseen_classes)
        Rs = class_prototypes(Xtc, Ytc, seen_classes)

        P = torch.cat([Ps, Pu], dim=0)
        source_classes = torch.cat([seen_classes, unseen_classes], dim=0)
        y_source = torch.cat([Ysc, Ysu], dim=0)

        Aohs = one_hot_prototype_matrix(y_source, source_classes, P)
        Aoht = one_hot_prototype_matrix(Ytc, seen_classes, Rs)
        Aoh = torch.cat([Aohs, Aoht], dim=0)

        M = (X - Aoh).T @ (X - Aoh)
        H = centering_matrix(n, device, dtype)

        A = X.T @ L @ X + self.alpha * M
        Bm = X.T @ H @ X
        A = 0.5 * (A + A.T)
        Bm = 0.5 * (Bm + Bm.T)

        p_eff = min(self.p, d)
        self.W, _ = solve_generalized_eig_top(A, Bm, p_eff, eps=self.eps)

        self.Ps, self.Pu, self.Rs = Ps, Pu, Rs
        self.seen_classes, self.unseen_classes = seen_classes, unseen_classes
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.W

    def prototypes_in_subspace(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.Ps @ self.W, self.Pu @ self.W, self.Rs @ self.W


# =============================================================================
# Stage 2 — prototype networks (prototype_nets.py)
# =============================================================================

Variant = Literal["full", "phi_only", "psi_only"]


class PhiNet(nn.Module):
    def __init__(self, p: int, m: int, final_relu: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(p, m, bias=False)
        self.fc2 = nn.Linear(m, p, bias=False)
        self.act = nn.ReLU()
        self.final_relu = final_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return self.act(out) if self.final_relu else out


class PsiNet(nn.Module):
    def __init__(self, num_seen: int, num_unseen: int, m: int, final_relu: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(num_seen, m, bias=False)
        self.fc2 = nn.Linear(m, num_unseen, bias=False)
        self.act = nn.ReLU()
        self.final_relu = final_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x.T))
        out = self.fc2(h)
        return self.act(out).T if self.final_relu else out.T


def fuse_unseen_prototypes(
    phi_out: Optional[torch.Tensor],
    psi_out: Optional[torch.Tensor],
    subspace_pwu: torch.Tensor,
    fusion: FusionMode,
) -> torch.Tensor:
    if fusion == "subspace":
        return subspace_pwu
    if fusion == "phi":
        if phi_out is None:
            raise ValueError("fusion='phi' requires the phi network.")
        return phi_out
    if fusion == "psi":
        if psi_out is None:
            raise ValueError("fusion='psi' requires the psi network.")
        return psi_out
    # fusion == "avg"
    if phi_out is None or psi_out is None:
        raise ValueError("fusion='avg' requires both phi and psi outputs.")
    return 0.5 * (phi_out + psi_out)


def train_prototype_networks(
    Pws: torch.Tensor,
    Pwu: torch.Tensor,
    Rws: torch.Tensor,
    m: int = 512,
    epochs: int = 2000,
    lr: float = 1e-4,
    variant: Variant = "full",
    fusion: FusionMode = "avg",
    final_relu: bool = False,
    loss_reduction: LossReduction = "sum",
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[Optional[nn.Module], Optional[nn.Module], torch.Tensor]:
    p = Pws.shape[1]
    C = Pws.shape[0]
    U = Pwu.shape[0]

    if fusion == "subspace":
        return None, None, Pwu.clone()

    phi = PhiNet(p, m, final_relu=final_relu).to(device) if variant != "psi_only" else None
    psi = PsiNet(C, U, m, final_relu=final_relu).to(device) if variant != "phi_only" else None

    params = []
    if phi is not None:
        params += list(phi.parameters())
    if psi is not None:
        params += list(psi.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    loss_fn = lambda a, b: frobenius_squared_loss(a, b, reduction=loss_reduction)

    for epoch in range(epochs):
        opt.zero_grad()
        loss = torch.zeros((), device=device)

        if variant == "full":
            phi_Pws = phi(Pws)
            psi_Pws = psi(Pws)
            phi_Pwu = phi(Pwu)
            psi_Rws = psi(Rws)

            l_mlp1 = loss_fn(phi_Pws, Rws)
            l_mlp2 = loss_fn(psi_Pws, Pwu)
            l_s = loss_fn(psi_Rws, phi_Pwu)
            loss = l_mlp1 + l_mlp2 + l_s
        elif variant == "phi_only":
            phi_Pws = phi(Pws)
            loss = loss_fn(phi_Pws, Rws)
        elif variant == "psi_only":
            psi_Pws = psi(Pws)
            loss = loss_fn(psi_Pws, Pwu)
        else:
            raise ValueError(f"unknown variant {variant!r}")

        loss.backward()
        opt.step()

        if verbose and (epoch + 1) % 200 == 0:
            print(f"[stage2:{variant}] epoch {epoch + 1}/{epochs} loss={loss.item():.6f}")

    with torch.no_grad():
        phi_out = phi(Pwu) if phi is not None else None
        psi_out = psi(Rws) if psi is not None else None
        if variant == "phi_only":
            Rwu = phi_out
        elif variant == "psi_only":
            Rwu = psi_out
        else:
            Rwu = fuse_unseen_prototypes(phi_out, psi_out, Pwu, fusion)

    return phi, psi, Rwu


# =============================================================================
# Data loading (datasets.py)
# =============================================================================

@dataclass
class DatasetDetails:
    prefix: str
    suffix: str
    resnet_feature: str
    split_file_name: str


_FEATURE_KEY_FALLBACKS = [
    "resnet50", "resnet50_features", "resnet101", "resnet101_features",
    "vgg16", "vgg16_features", "clip", "clip_features",
    "features", "feature", "X",
]
_LABEL_KEY_FALLBACKS = ["labels", "label", "y", "Y"]


def _lookup(mat_dict: dict, preferred_key: str, fallbacks: list, file_path: str):
    tried = [preferred_key] + [k for k in fallbacks if k != preferred_key]
    for k in tried:
        if k in mat_dict:
            return mat_dict[k], k
    available = sorted(k for k in mat_dict.keys() if not k.startswith("__"))
    raise KeyError(
        f"Could not find any of {tried} in {file_path!r}.\n"
        f"Available keys in this file: {available}\n"
        f"Fix `dataset_details['resnet_feature']` (or the label key, if "
        f"that's the one missing) accordingly, or run:\n"
        f"    python src/tupl.py inspect {file_path}\n"
        f"to see the full structure."
    )


class TUPLDataset:
    def __init__(
        self,
        domain_set: List[str],
        data_dir: str,
        source_domain_index: int,
        target_domain_index: int,
        trial_index: int,
        dataset_details: Dict,
        device: str = "cpu",
    ):
        self.domain_set = domain_set
        self.data_dir = data_dir
        self.dataset_details = dataset_details
        self.device = device
        self._load_mat(source_domain_index, target_domain_index, trial_index)

    def _load_mat(self, source_domain_index, target_domain_index, trial_index):
        dd = self.dataset_details

        path_A = self.data_dir + dd["prefix"] + self.domain_set[source_domain_index] + dd["suffix"]
        data_A = scipy.io.loadmat(path_A)
        feature_A, _ = _lookup(data_A, dd["resnet_feature"], _FEATURE_KEY_FALLBACKS, path_A)
        feature_A = normalize(feature_A.squeeze(), norm="l2")
        label_A_raw, _ = _lookup(data_A, "labels", _LABEL_KEY_FALLBACKS, path_A)
        label_A = label_A_raw.squeeze().astype(np.int64).ravel()

        path_B = self.data_dir + dd["prefix"] + self.domain_set[target_domain_index] + dd["suffix"]
        data_B = scipy.io.loadmat(path_B)
        feature_B, _ = _lookup(data_B, dd["resnet_feature"], _FEATURE_KEY_FALLBACKS, path_B)
        feature_B = normalize(feature_B.squeeze(), norm="l2")
        label_B_raw, _ = _lookup(data_B, "labels", _LABEL_KEY_FALLBACKS, path_B)
        label_B = label_B_raw.squeeze().astype(np.int64).ravel()

        split_path = self.data_dir + dd["split_file_name"]
        data_split = scipy.io.loadmat(split_path)
        try:
            split_flag_B = data_split["targetDomain_splitFlag"][0, trial_index][
                0, target_domain_index
            ][0,]
            unseen_classes = data_split["targetDomain_unseenClass"][0, trial_index][
                0, target_domain_index
            ][0,]
        except (KeyError, IndexError) as e:
            available = sorted(k for k in data_split.keys() if not k.startswith("__"))
            raise type(e)(
                f"Failed to index split file {split_path!r} with "
                f"trial_index={trial_index}, target_domain_index={target_domain_index}: {e}\n"
                f"Available keys: {available}\n"
                f"Run `python src/tupl.py inspect {split_path}` to inspect its structure."
            ) from e

        unseen_indicator = np.asarray(unseen_classes).astype(np.int64).ravel()
        all_classes = np.unique(label_A)

        if len(unseen_indicator) != len(all_classes):
            raise ValueError(
                f"Unseen-class indicator length ({len(unseen_indicator)}) does not "
                f"match number of classes ({len(all_classes)}). "
                f"Expected a per-class 0/1 mask (0=seen, 1=unseen), as in "
                f"BaseTwoModalDataset.unseenClass_B."
            )
        seen_classes = all_classes[unseen_indicator == 0]
        unseen_classes = all_classes[unseen_indicator == 1]

        expected_seen = dd.get("num_seen")
        expected_unseen = dd.get("num_unseen")
        if expected_seen is not None and len(seen_classes) != expected_seen:
            raise ValueError(
                f"Expected {expected_seen} seen classes but found {len(seen_classes)}."
            )
        if expected_unseen is not None and len(unseen_classes) != expected_unseen:
            raise ValueError(
                f"Expected {expected_unseen} unseen classes but found {len(unseen_classes)}."
            )

        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.num_seen = len(seen_classes)
        self.num_unseen = len(unseen_classes)
        self.feature_dim = feature_A.shape[1]

        is_unseen_A = np.isin(label_A, unseen_classes)
        self.Xsc = feature_A[~is_unseen_A]
        self.Ysc = label_A[~is_unseen_A]
        self.Xsu = feature_A[is_unseen_A]
        self.Ysu = label_A[is_unseen_A]

        assert len(self.Xsc) > 0 and len(self.Xsu) > 0, (
            "Source domain must contain both seen and unseen class samples "
            "(GZSDA requires all classes to exist in the source domain)."
        )

        self.Xtc = feature_B[split_flag_B == 1]
        self.Ytc = label_B[split_flag_B == 1]
        self.Xte = feature_B[split_flag_B == 2]
        self.Yte = label_B[split_flag_B == 2]

        assert np.isin(self.Ytc, seen_classes).all(), (
            "Target training samples (splitFlag == 1) must all belong to "
            "seen classes, per the GZSDA setting."
        )

    def as_tensors(self, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        dev = device or self.device
        to_f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
        to_l = lambda x: torch.as_tensor(x, dtype=torch.long, device=dev)
        return dict(
            Xsc=to_f(self.Xsc),
            Ysc=to_l(self.Ysc),
            Xsu=to_f(self.Xsu),
            Ysu=to_l(self.Ysu),
            Xtc=to_f(self.Xtc),
            Ytc=to_l(self.Ytc),
            Xte=to_f(self.Xte),
            Yte=to_l(self.Yte),
            seen_classes=to_l(self.seen_classes),
            unseen_classes=to_l(self.unseen_classes),
        )

    def summary(self) -> str:
        return (
            f"seen classes: {self.num_seen}  unseen classes: {self.num_unseen}\n"
            f"source seen samples:   {len(self.Xsc):6d}\n"
            f"source unseen samples: {len(self.Xsu):6d}\n"
            f"target train (seen):   {len(self.Xtc):6d}\n"
            f"target test (seen+unseen): {len(self.Xte):6d}"
        )


class TargetTestDataset(Dataset):
    def __init__(self, Xte: torch.Tensor, Yte: torch.Tensor):
        self.Xte = Xte
        self.Yte = Yte

    def __len__(self):
        return self.Xte.shape[0]

    def __getitem__(self, idx):
        return self.Xte[idx], self.Yte[idx]


# =============================================================================
# End-to-end TUPL model (tupl.py)
# =============================================================================

class TUPL:
    def __init__(
        self,
        p: int = 1024,
        alpha: float = 1.0,
        sigma: Optional[float] = None,
        kernel_mode: GraphKernelMode = "paper",
        m: int = 512,
        stage2_epochs: int = 2000,
        stage2_lr: float = 1e-4,
        stage2_variant: Variant = "full",
        fusion: FusionMode = "avg",
        final_relu: bool = False,
        loss_reduction: LossReduction = "sum",
        metric: MetricMode = "macro",
        device: str = "cpu",
    ):
        self.subspace = SubspaceLearner(
            p=p, alpha=alpha, sigma=sigma, kernel_mode=kernel_mode
        )
        self.m = m
        self.stage2_epochs = stage2_epochs
        self.stage2_lr = stage2_lr
        self.stage2_variant = stage2_variant
        self.fusion = fusion
        self.final_relu = final_relu
        self.loss_reduction = loss_reduction
        self.metric = metric
        self.device = device
        self.Pwu_subspace: Optional[torch.Tensor] = None

    def fit(self, data: Dict[str, torch.Tensor], verbose: bool = False) -> "TUPL":
        Xsc, Ysc = data["Xsc"], data["Ysc"]
        Xsu, Ysu = data["Xsu"], data["Ysu"]
        Xtc, Ytc = data["Xtc"], data["Ytc"]
        seen_classes, unseen_classes = data["seen_classes"], data["unseen_classes"]

        self.subspace.fit(Xsc, Ysc, Xsu, Ysu, Xtc, Ytc, seen_classes, unseen_classes)
        Pws, Pwu, Rws = self.subspace.prototypes_in_subspace()
        self.Pwu_subspace = Pwu

        self.phi, self.psi, Rwu = train_prototype_networks(
            Pws,
            Pwu,
            Rws,
            m=self.m,
            epochs=self.stage2_epochs,
            lr=self.stage2_lr,
            variant=self.stage2_variant,
            fusion=self.fusion,
            final_relu=self.final_relu,
            loss_reduction=self.loss_reduction,
            device=self.device,
            verbose=verbose,
        )

        self.Rws = Rws
        self.Rwu = Rwu
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.all_classes = torch.cat([seen_classes, unseen_classes], dim=0)
        self.all_prototypes = torch.cat([Rws, Rwu], dim=0)
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        Xw = self.subspace.transform(X)
        return nearest_prototype_classify(
            Xw, self.all_prototypes, self.all_classes, batch_size
        )

    def evaluate(
        self,
        Xte: torch.Tensor,
        Yte: torch.Tensor,
        batch_size: Optional[int] = None,
        metric: Optional[MetricMode] = None,
    ):
        y_pred = self.predict(Xte, batch_size=batch_size)
        return harmonic_mean_accuracy(
            Yte,
            y_pred,
            self.seen_classes,
            self.unseen_classes,
            metric=metric or self.metric,
        )


# =============================================================================
# Data validation & stage diagnostics
# =============================================================================

def validate_dataset_files(
    dataset: str,
    data_root: str = "../data/",
    backbone: Optional[str] = None,
) -> Dict[str, object]:
    """Validate that feature and split files exist and match expected schema."""
    backbone = backbone or next(iter(DATASET_CONFIGS[dataset]["backbones"]))
    domain_set, data_dir, dd = get_dataset_details(dataset, backbone, data_root=data_root)

    missing = []
    for domain in domain_set:
        path = data_dir + dd["prefix"] + domain + dd["suffix"]
        if not os.path.isfile(path):
            missing.append(path)

    split_path = data_dir + dd["split_file_name"]
    if not os.path.isfile(split_path):
        missing.append(split_path)

    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s):\n" + "\n".join(f"  - {p}" for p in missing)
        )

    sample_path = data_dir + dd["prefix"] + domain_set[0] + dd["suffix"]
    sample = scipy.io.loadmat(sample_path)
    feat, feat_key = _lookup(
        sample, dd["resnet_feature"], _FEATURE_KEY_FALLBACKS, sample_path
    )
    labels, _ = _lookup(sample, "labels", _LABEL_KEY_FALLBACKS, sample_path)
    split = scipy.io.loadmat(split_path)
    n_trials = int(split["targetDomain_splitFlag"].shape[1])

    report = {
        "dataset": dataset,
        "backbone": backbone,
        "data_dir": data_dir,
        "domains": domain_set,
        "feature_key": feat_key,
        "feature_shape": tuple(feat.squeeze().shape),
        "label_shape": tuple(labels.shape),
        "num_trials": n_trials,
        "split_file": split_path,
    }
    return report


def diagnose_stages(
    data: Dict[str, torch.Tensor],
    model: TUPL,
    metric: MetricMode = "macro",
) -> Dict[str, object]:
    """Per-stage diagnostics for unseen prototype learning and classification."""
    seen = model.seen_classes
    unseen = model.unseen_classes
    Xte, Yte = data["Xte"], data["Yte"]
    unseen_mask = torch.isin(Yte, unseen)

    Pws, Pwu, Rws = model.subspace.prototypes_in_subspace()
    Xw = model.subspace.transform(Xte)

    def _eval(Rwu: torch.Tensor, label: str) -> Dict[str, float]:
        all_p = torch.cat([Rws, Rwu], dim=0)
        all_c = torch.cat([seen, unseen], dim=0)
        preds = nearest_prototype_classify(Xw, all_p, all_c)
        acc_s, acc_u, h = harmonic_mean_accuracy(Yte, preds, seen, unseen, metric=metric)
        unseen_preds_seen = (
            torch.isin(preds[unseen_mask], seen).sum().item() if unseen_mask.any() else 0
        )
        return {
            "label": label,
            "Accs": acc_s,
            "Accu": acc_u,
            "H": h,
            "Rwu_norm": float(Rwu.norm(dim=1).mean().item()),
            "unseen_test_as_seen": int(unseen_preds_seen),
            "unseen_test_count": int(unseen_mask.sum().item()),
        }

    stages: List[Dict[str, object]] = []
    stages.append(_eval(Pwu, "stage1_subspace_Pwu"))

    with torch.no_grad():
        if model.phi is not None:
            stages.append(_eval(model.phi(Pwu), "stage2_phi_Pwu"))
        if model.psi is not None:
            stages.append(_eval(model.psi(Rws), "stage2_psi_Rws"))
        stages.append(_eval(model.Rwu, "final_Rwu"))

    final_preds = model.predict(Xte)
    unseen_pred_counts = {
        int(c.item()): int((final_preds[unseen_mask] == c).sum().item())
        for c in unseen
    }
    return {
        "metric": metric,
        "stages": stages,
        "unseen_prediction_histogram": unseen_pred_counts,
    }


def export_results(
    results: Sequence[Dict],
    output_dir: str,
    prefix: str = "tupl",
    group_by: Sequence[str] = ("source_name", "target_name"),
) -> Tuple[str, str]:
    """Write raw run results and aggregated pair summaries to JSON and CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_json = out / f"{prefix}_raw.json"
    raw_csv = out / f"{prefix}_raw.csv"
    summary_json = out / f"{prefix}_summary.json"
    summary_csv = out / f"{prefix}_summary.csv"

    with raw_json.open("w", encoding="utf-8") as f:
        json.dump(list(results), f, indent=2)

    if results:
        fieldnames = list(results[0].keys())
        with raw_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    pair_summary = summarize_pair_results(results, group_by=group_by)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(pair_summary, f, indent=2)

    if pair_summary:
        summary_fields = list(pair_summary[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(pair_summary)

    return str(raw_json), str(summary_json)


def _mean_sem(values: np.ndarray) -> Tuple[float, float]:
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem


def prepare_report(results: Sequence[Dict[str, float]]) -> str:
    """
    Mean ± SEM over trials for one scenario (domain pair).

    Matches refactored ``src/utils.py`` ``prepare_report``: average per-trial
    seen/unseen accuracies (percent), compute H per trial, then average.
    """
    seen = np.array([r["Accs"] for r in results], dtype=np.float64)
    unseen = np.array([r["Accu"] for r in results], dtype=np.float64)
    h = seen * unseen * 2 / (seen + unseen)

    mean_seen, sem_seen = _mean_sem(seen)
    mean_unseen, sem_unseen = _mean_sem(unseen)
    mean_h, sem_h = _mean_sem(h)

    return (
        f"Seen:     {mean_seen:.2f} ± {sem_seen:.2f}\n"
        f"Unseen:   {mean_unseen:.2f} ± {sem_unseen:.2f}\n"
        f"H-mean:   {mean_h:.2f} ± {sem_h:.2f}"
    )


def summarize_pair_results(
    results: Sequence[Dict],
    group_by: Sequence[str] = ("source_name", "target_name"),
) -> List[Dict]:
    """Mean ± SEM of Accs/Accu/H grouped by domain pair (trial-averaged per scenario)."""
    buckets: Dict[Tuple, List[Dict]] = {}
    for row in results:
        key = tuple(row[k] for k in group_by)
        buckets.setdefault(key, []).append(row)

    summary = []
    for key, rows in sorted(buckets.items()):
        entry = {group_by[i]: key[i] for i in range(len(group_by))}
        entry["n_trials"] = len(rows)
        seen = np.array([r["Accs"] for r in rows], dtype=np.float64)
        unseen = np.array([r["Accu"] for r in rows], dtype=np.float64)
        h = seen * unseen * 2 / (seen + unseen)
        entry["Accs_mean"], entry["Accs_sem"] = _mean_sem(seen)
        entry["Accu_mean"], entry["Accu_sem"] = _mean_sem(unseen)
        entry["H_mean"], entry["H_sem"] = _mean_sem(h)
        pair = f"{entry.get('source_name', '')} -> {entry.get('target_name', '')}"
        if pair in PAPER_OFFICE31_H_TARGETS:
            entry["paper_H"] = PAPER_OFFICE31_H_TARGETS[pair]
            entry["H_gap"] = entry["H_mean"] - entry["paper_H"]
        summary.append(entry)
    return summary


def format_summary_row_report(row: Dict) -> str:
    """Format one pair summary like refactored prepare_report output."""
    return (
        f"Seen:     {row['Accs_mean']:.2f} ± {row['Accs_sem']:.2f}\n"
        f"Unseen:   {row['Accu_mean']:.2f} ± {row['Accu_sem']:.2f}\n"
        f"H-mean:   {row['H_mean']:.2f} ± {row['H_sem']:.2f}"
    )


def summary_to_report_map(
    summary: Sequence[Dict],
    method: str = "TUPL",
) -> Dict[str, Dict[str, str]]:
    """Convert pair summaries to refactored-style nested dict."""
    pair_reports = {
        f"{row['source_name']} -> {row['target_name']}": format_summary_row_report(row)
        for row in summary
    }
    return {method: pair_reports}


REFACTORED_CSV_FIELDNAMES = ["domain", "method", "seen", "unseen", "H-mean"]
REFACTORED_METHOD_ORDER = ("base", "CCVAE", "our0", "our_GRE", "TUPL")


def _normalize_refactored_csv_row(row: Dict) -> Dict[str, str]:
    """Keep only refactored CSV columns (drop pandas index / stray columns)."""
    return {key: row[key] for key in REFACTORED_CSV_FIELDNAMES if key in row}


def _method_sort_key(method: str) -> Tuple[int, str]:
    try:
        return (REFACTORED_METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(REFACTORED_METHOD_ORDER), method)


def _sort_refactored_csv_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Group by domain pair, then method (same order as refactored-*.ipynb)."""
    return sorted(
        rows,
        key=lambda row: (row.get("domain", ""), _method_sort_key(row.get("method", ""))),
    )


def export_refactored_results(
    summary: Sequence[Dict],
    result_root: str = "../result",
    dataset: str = "office31",
    method: str = "TUPL",
    merge_existing: bool = True,
) -> Tuple[str, str]:
    """
    Save TUPL results in the same layout as refactored-*.ipynb:
    `{result_root}/json/{name}.json` and `{result_root}/csv/{name}.csv`.
    """
    name = get_result_name(dataset)
    root = Path(result_root)
    json_path = root / "json" / f"{name}.json"
    csv_path = root / "csv" / f"{name}.csv"
    root.joinpath("json").mkdir(parents=True, exist_ok=True)
    root.joinpath("csv").mkdir(parents=True, exist_ok=True)

    report_map = summary_to_report_map(summary, method=method)
    if merge_existing and json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(report_map)
        report_map = existing

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_map, f, indent=2)

    csv_rows: List[Dict[str, str]] = []
    if merge_existing and csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = [
                _normalize_refactored_csv_row(row)
                for row in reader
                if row.get("method") != method and row.get("domain")
            ]

    for row in summary:
        pair = f"{row['source_name']} -> {row['target_name']}"
        csv_rows.append(
            {
                "domain": pair,
                "method": method,
                "seen": f"{row['Accs_mean']:.2f} ± {row['Accs_sem']:.2f}",
                "unseen": f"{row['Accu_mean']:.2f} ± {row['Accu_sem']:.2f}",
                "H-mean": f"{row['H_mean']:.2f} ± {row['H_sem']:.2f}",
            }
        )

    csv_rows = _sort_refactored_csv_rows(csv_rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REFACTORED_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(csv_rows)

    return str(json_path), str(csv_path)


def sort_refactored_result_csvs(result_root: str = "../result") -> List[str]:
    """Rewrite `{result_root}/csv/*.csv` grouped by domain, then method."""
    csv_dir = Path(result_root) / "csv"
    rewritten: List[str] = []
    if not csv_dir.is_dir():
        return rewritten
    for csv_path in sorted(csv_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8") as f:
            rows = [
                _normalize_refactored_csv_row(row)
                for row in csv.DictReader(f)
                if row.get("domain")
            ]
        rows = _sort_refactored_csv_rows(rows)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REFACTORED_CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        rewritten.append(str(csv_path))
    return rewritten


# =============================================================================
# .mat inspector (inspect_mat.py)
# =============================================================================

def _describe_mat_value(value, indent=0):
    pad = "  " * indent
    if isinstance(value, np.ndarray):
        print(f"{pad}ndarray  shape={value.shape}  dtype={value.dtype}")
        if value.dtype == object and value.size > 0 and value.size <= 8:
            flat = value.ravel()
            for i, v in enumerate(flat):
                print(f"{pad}  [{i}] ->")
                _describe_mat_value(v, indent + 2)
        elif value.dtype == object and value.size > 0:
            print(f"{pad}  (showing element [0] of {value.size})")
            _describe_mat_value(value.ravel()[0], indent + 1)
    else:
        print(f"{pad}{type(value)} = {value!r}")


def inspect_mat(path: str) -> None:
    mat = scipy.io.loadmat(path)
    print(f"file: {path}\n")
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        print(f"key: {key!r}")
        _describe_mat_value(value, indent=1)
        print()


# =============================================================================
# Synthetic smoke test (test_synthetic.py)
# =============================================================================

def make_synthetic_data(
    num_seen=6, num_unseen=4, d=64, n_per_class=30, domain_shift=0.5
) -> Dict[str, torch.Tensor]:
    seen_classes = torch.arange(num_seen)
    unseen_classes = torch.arange(num_seen, num_seen + num_unseen)
    centers = torch.randn(num_seen + num_unseen, d) * 4.0

    def sample(classes, n_per_class, shift):
        Xs, Ys = [], []
        for c in classes:
            mean = centers[c] + shift
            X = mean + torch.randn(n_per_class, d) * 0.5
            Xs.append(X)
            Ys.append(torch.full((n_per_class,), int(c.item()), dtype=torch.long))
        return torch.cat(Xs), torch.cat(Ys)

    Xsc, Ysc = sample(seen_classes, n_per_class, shift=0.0)
    Xsu, Ysu = sample(unseen_classes, n_per_class, shift=0.0)
    Xtc, Ytc = sample(seen_classes, n_per_class, shift=domain_shift)
    Xte_seen, Yte_seen = sample(seen_classes, 15, shift=domain_shift)
    Xte_unseen, Yte_unseen = sample(unseen_classes, 15, shift=domain_shift)
    Xte = torch.cat([Xte_seen, Xte_unseen])
    Yte = torch.cat([Yte_seen, Yte_unseen])

    return dict(
        Xsc=Xsc,
        Ysc=Ysc,
        Xsu=Xsu,
        Ysu=Ysu,
        Xtc=Xtc,
        Ytc=Ytc,
        Xte=Xte,
        Yte=Yte,
        seen_classes=seen_classes,
        unseen_classes=unseen_classes,
    )


def run_synthetic_test(
    variant: Variant = "full",
    p: int = 32,
    alpha: float = 1.0,
    m: int = 32,
    stage2_epochs: int = 500,
    stage2_lr: float = 1e-3,
    seed: int = 0,
    fusion: FusionMode = "avg",
    final_relu: bool = False,
    metric: MetricMode = "macro",
    verbose: bool = False,
):
    set_seed(seed)
    data = make_synthetic_data()
    model = TUPL(
        p=p,
        alpha=alpha,
        m=m,
        stage2_epochs=stage2_epochs,
        stage2_lr=stage2_lr,
        stage2_variant=variant,
        fusion=fusion,
        final_relu=final_relu,
        metric=metric,
    )
    model.fit(data, verbose=verbose)
    acc_s, acc_u, h = model.evaluate(data["Xte"], data["Yte"])
    print(f"variant={variant:9s} Accs={acc_s:5.1f}  Accu={acc_u:5.1f}  H={h:5.1f}")
    return acc_s, acc_u, h


def run_regression_tests(seed: int = 0) -> bool:
    """Fast regression checks for graph, metrics, determinism, and unseen predictions."""
    set_seed(seed)
    ok = True

    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = torch.tensor([0, 0, 1, 1])
    Q, L = build_similarity_graph(X, y, kernel_mode="paper")
    if not torch.isfinite(Q).all() or not torch.isfinite(L).all():
        print("FAIL: similarity graph produced non-finite values")
        ok = False
    if Q[0, 1] <= 0 or Q[2, 3] <= 0:
        print("FAIL: same-class pairs should have positive similarity")
        ok = False

    y_true = torch.tensor([0, 0, 1, 1, 2, 2])
    y_pred = torch.tensor([0, 1, 1, 1, 2, 0])
    seen = torch.tensor([0, 1])
    unseen = torch.tensor([2])
    macro_s, macro_u, _ = macro_per_class_accuracy(y_true, y_pred, seen, unseen)
    if abs(macro_s - 75.0) > 1e-6 or abs(macro_u - 50.0) > 1e-6:
        print(f"FAIL: macro metrics expected (75, 50) got ({macro_s}, {macro_u})")
        ok = False

    set_seed(seed)
    acc_s, acc_u, h = run_synthetic_test(
        variant="full", p=32, m=32, stage2_epochs=300, final_relu=False, metric="macro"
    )
    if acc_u < 50.0 or h < 50.0:
        print(f"FAIL: synthetic metrics too low (Accs={acc_s:.1f}, Accu={acc_u:.1f}, H={h:.1f})")
        ok = False

    if ok:
        print("All regression tests passed.")
    return ok


# =============================================================================
# Training entry point (train.py)
# =============================================================================

def run_tupl(
    data_root: str = "../data/",
    dataset: str = "officehome",
    backbone: Optional[str] = None,
    source: int = 0,
    target: int = 1,
    trial: int = 0,
    p: int = 1024,
    alpha: float = 1.0,
    sigma: Optional[float] = None,
    kernel_mode: GraphKernelMode = "paper",
    m: int = 512,
    stage2_epochs: int = 2000,
    stage2_lr: float = 1e-4,
    variant: Variant = "full",
    fusion: FusionMode = "avg",
    final_relu: bool = False,
    loss_reduction: LossReduction = "sum",
    metric: MetricMode = "macro",
    seed: Optional[int] = 1,
    verbose: bool = False,
    device: Optional[str] = None,
    quiet: bool = False,
    return_model: bool = True,
):
    if seed is not None:
        set_seed(seed)

    backbone = backbone or next(iter(DATASET_CONFIGS[dataset]["backbones"]))
    domain_set, data_dir, dataset_details = get_dataset_details(
        dataset, backbone, data_root=data_root
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = TUPLDataset(
        domain_set=domain_set,
        data_dir=data_dir,
        source_domain_index=source,
        target_domain_index=target,
        trial_index=trial,
        dataset_details=dataset_details,
        device=device,
    )
    if not quiet:
        print(f"[{dataset}/{backbone}] {domain_set[source]} -> "
              f"{domain_set[target]} (trial {trial})")
        print(ds.summary())

    data = ds.as_tensors(device=device)
    model = TUPL(
        p=p,
        alpha=alpha,
        sigma=sigma,
        kernel_mode=kernel_mode,
        m=m,
        stage2_epochs=stage2_epochs,
        stage2_lr=stage2_lr,
        stage2_variant=variant,
        fusion=fusion,
        final_relu=final_relu,
        loss_reduction=loss_reduction,
        metric=metric,
        device=device,
    )
    model.fit(data, verbose=verbose)

    acc_s, acc_u, h = model.evaluate(data["Xte"], data["Yte"])
    if not quiet:
        print(f"Accs={acc_s:.1f}  Accu={acc_u:.1f}  H={h:.1f}  (metric={metric})")

    metrics = (acc_s, acc_u, h)
    if return_model:
        return model, data, metrics
    return metrics


def run_dataset_reproduction(
    dataset: str,
    data_root: str = "../data/",
    output_dir: str = "results",
    backbone: Optional[str] = None,
    result_root: Optional[str] = "../result",
    seed: int = 1,
    device: Optional[str] = None,
    fusion: FusionMode = "avg",
    metric: MetricMode = "macro",
    p: int = 1024,
    alpha: float = 1.0,
    kernel_mode: GraphKernelMode = "paper",
    m: int = 512,
    stage2_epochs: int = 2000,
    stage2_lr: float = 1e-4,
    variant: Variant = "full",
    quiet: bool = False,
    export_refactored: bool = True,
    verbose_trials: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """Run all domain pairs × all trials for one GZSDA dataset."""
    backbone = backbone or get_default_backbone(dataset)
    validate_dataset_files(dataset, data_root=data_root, backbone=backbone)
    domain_set, _, _ = get_dataset_details(dataset, backbone, data_root=data_root)
    trials = list(range(get_num_trials(dataset, data_root, backbone)))
    pairs = list(iter_domain_pairs(domain_set))

    results: List[Dict] = []
    total = len(pairs) * len(trials)
    run_idx = 0
    for source, target in pairs:
        scenario = f"{domain_set[source]} -> {domain_set[target]}"
        trial_rows: List[Dict] = []
        for trial in trials:
            run_idx += 1
            if verbose_trials and not quiet:
                print(
                    f"[{dataset}] [{run_idx}/{total}] {scenario} trial={trial}"
                )
            acc_s, acc_u, h = run_tupl(
                data_root=data_root,
                dataset=dataset,
                backbone=backbone,
                source=source,
                target=target,
                trial=trial,
                p=p,
                alpha=alpha,
                kernel_mode=kernel_mode,
                m=m,
                stage2_epochs=stage2_epochs,
                stage2_lr=stage2_lr,
                variant=variant,
                fusion=fusion,
                final_relu=False,
                loss_reduction="sum",
                metric=metric,
                seed=seed,
                device=device,
                quiet=True,
                return_model=False,
            )
            row = {
                "dataset": dataset,
                "backbone": backbone,
                "source": source,
                "target": target,
                "source_name": domain_set[source],
                "target_name": domain_set[target],
                "trial": trial,
                "variant": variant,
                "fusion": fusion,
                "metric": metric,
                "p": p,
                "alpha": alpha,
                "kernel_mode": kernel_mode,
                "m": m,
                "stage2_epochs": stage2_epochs,
                "stage2_lr": stage2_lr,
                "final_relu": False,
                "loss_reduction": "sum",
                "seed": seed,
                "Accs": acc_s,
                "Accu": acc_u,
                "H": h,
            }
            trial_rows.append(row)
            results.append(row)

        if not quiet:
            print(scenario)
            print(prepare_report(trial_rows))

    prefix = get_result_name(dataset)
    export_results(results, output_dir, prefix=prefix)
    summary = summarize_pair_results(results)
    if export_refactored and result_root is not None:
        export_refactored_results(summary, result_root=result_root, dataset=dataset)

    if not quiet:
        print(f"\n=== {dataset}: {len(pairs)} scenarios × {len(trials)} trials ===")
    return results, summary


def run_all_datasets_reproduction(
    datasets: Optional[Sequence[str]] = None,
    data_root: str = "../data/",
    output_dir: str = "results",
    result_root: str = "../result",
    **kwargs,
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """Run TUPL on every configured dataset (like separate refactored-*.ipynb)."""
    datasets = list(datasets or DATASET_CONFIGS.keys())
    all_results: Dict[str, Tuple[List[Dict], List[Dict]]] = {}
    for dataset in datasets:
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASET_CONFIGS)}")
        print(f"\n{'=' * 60}\nRunning TUPL on {dataset}\n{'=' * 60}")
        all_results[dataset] = run_dataset_reproduction(
            dataset=dataset,
            data_root=data_root,
            output_dir=output_dir,
            result_root=result_root,
            **kwargs,
        )
    return all_results


def run_office31_reproduction(
    data_root: str = "../data/",
    output_dir: str = "results",
    seed: int = 1,
    device: Optional[str] = None,
    fusion: FusionMode = "avg",
    metric: MetricMode = "macro",
    quiet: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """Run official Office31 protocol: 6 pairs × 5 trials with paper defaults."""
    return run_dataset_reproduction(
        dataset="office31",
        data_root=data_root,
        output_dir=output_dir,
        backbone="resnet50",
        result_root="../result",
        seed=seed,
        device=device,
        fusion=fusion,
        metric=metric,
        quiet=quiet,
    )


def run_experiment_grid(
    data_root: str = "../data/",
    dataset: str = "officehome",
    backbone: Optional[str] = None,
    trials: Optional[Sequence[int]] = None,
    source_targets: Optional[Sequence[Tuple[int, int]]] = None,
    variants: Sequence[Variant] = ("full",),
    fusion_modes: Sequence[FusionMode] = ("avg",),
    ps: Sequence[int] = (1024,),
    alphas: Sequence[float] = (1.0,),
    kernel_modes: Sequence[GraphKernelMode] = ("paper",),
    ms: Sequence[int] = (512,),
    stage2_epochs_list: Sequence[int] = (2000,),
    stage2_lrs: Sequence[float] = (1e-4,),
    final_relu: bool = False,
    loss_reduction: LossReduction = "sum",
    metric: MetricMode = "macro",
    seed: Optional[int] = 1,
    device: Optional[str] = None,
    verbose: bool = False,
    quiet: bool = True,
    on_progress: Optional[Callable[[int, int, int, int, int, str], None]] = None,
) -> List[Dict]:
    """
    Run TUPL over a Cartesian grid of episodes and hyperparameters.

    Defaults sweep all trials and all source->target domain pairs for
    `dataset`. Returns one dict per run with metrics and run settings.
    """
    backbone = backbone or next(iter(DATASET_CONFIGS[dataset]["backbones"]))
    domain_set, _, _ = get_dataset_details(dataset, backbone, data_root=data_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if trials is None:
        trials = list(range(get_num_trials(dataset, data_root, backbone)))
    if source_targets is None:
        source_targets = list(iter_domain_pairs(domain_set))

    hyperparams = list(
        itertools.product(
            variants, fusion_modes, ps, alphas, kernel_modes, ms,
            stage2_epochs_list, stage2_lrs,
        )
    )
    episodes = list(itertools.product(source_targets, trials))
    total = len(episodes) * len(hyperparams)
    results: List[Dict] = []
    run_idx = 0

    for (source, target), trial in episodes:
        src_name = domain_set[source]
        tgt_name = domain_set[target]
        for variant, fusion, p, alpha, kernel_mode, m, stage2_epochs, stage2_lr in hyperparams:
            run_idx += 1
            if on_progress is not None:
                on_progress(run_idx, total, source, target, trial, variant)
            elif not quiet:
                print(
                    f"[{run_idx}/{total}] {src_name}->{tgt_name} trial={trial} "
                    f"variant={variant} fusion={fusion} p={p} alpha={alpha} "
                    f"kernel={kernel_mode} m={m} epochs={stage2_epochs} lr={stage2_lr}"
                )

            acc_s, acc_u, h = run_tupl(
                data_root=data_root,
                dataset=dataset,
                backbone=backbone,
                source=source,
                target=target,
                trial=trial,
                p=p,
                alpha=alpha,
                kernel_mode=kernel_mode,
                m=m,
                stage2_epochs=stage2_epochs,
                stage2_lr=stage2_lr,
                variant=variant,
                fusion=fusion,
                final_relu=final_relu,
                loss_reduction=loss_reduction,
                metric=metric,
                seed=seed,
                verbose=verbose,
                device=device,
                quiet=True,
                return_model=False,
            )
            results.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "source": source,
                    "target": target,
                    "source_name": src_name,
                    "target_name": tgt_name,
                    "trial": trial,
                    "variant": variant,
                    "fusion": fusion,
                    "metric": metric,
                    "p": p,
                    "alpha": alpha,
                    "kernel_mode": kernel_mode,
                    "m": m,
                    "stage2_epochs": stage2_epochs,
                    "stage2_lr": stage2_lr,
                    "final_relu": final_relu,
                    "loss_reduction": loss_reduction,
                    "seed": seed,
                    "Accs": acc_s,
                    "Accu": acc_u,
                    "H": h,
                }
            )

    return results


def summarize_results(
    results: Sequence[Dict],
    group_by: Sequence[str] = ("variant", "alpha"),
) -> List[Dict]:
    """Mean Accs/Accu/H over runs, grouped by the given keys."""
    buckets: Dict[Tuple, List[Dict]] = {}
    for row in results:
        key = tuple(row[k] for k in group_by)
        buckets.setdefault(key, []).append(row)

    summary = []
    for key, rows in sorted(buckets.items()):
        entry = {group_by[i]: key[i] for i in range(len(group_by))}
        entry["n_runs"] = len(rows)
        entry["Accs_mean"] = float(np.mean([r["Accs"] for r in rows]))
        entry["Accu_mean"] = float(np.mean([r["Accu"] for r in rows]))
        entry["H_mean"] = float(np.mean([r["H"] for r in rows]))
        summary.append(entry)
    return summary


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "inspect":
        if len(sys.argv) != 3:
            print("Usage: python src/tupl.py inspect <path-to-mat>")
            sys.exit(1)
        inspect_mat(sys.argv[2])
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "synthetic":
        for variant in ["full", "phi_only", "psi_only"]:
            run_synthetic_test(variant=variant)
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "regression":
        ok = run_regression_tests()
        sys.exit(0 if ok else 1)

    if len(sys.argv) >= 2 and sys.argv[1] == "office31":
        data_root = "../data/"
        output_dir = "results"
        if "--data_root" in sys.argv:
            data_root = sys.argv[sys.argv.index("--data_root") + 1]
        if "--output_dir" in sys.argv:
            output_dir = sys.argv[sys.argv.index("--output_dir") + 1]
        run_office31_reproduction(data_root=data_root, output_dir=output_dir)
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "all":
        data_root = "../data/"
        output_dir = "results"
        result_root = "../result"
        datasets = None
        if "--data_root" in sys.argv:
            data_root = sys.argv[sys.argv.index("--data_root") + 1]
        if "--output_dir" in sys.argv:
            output_dir = sys.argv[sys.argv.index("--output_dir") + 1]
        if "--result_root" in sys.argv:
            result_root = sys.argv[sys.argv.index("--result_root") + 1]
        if "--datasets" in sys.argv:
            datasets = [
                d.strip() for d in sys.argv[sys.argv.index("--datasets") + 1].split(",") if d.strip()
            ]
        run_all_datasets_reproduction(
            datasets=datasets,
            data_root=data_root,
            output_dir=output_dir,
            result_root=result_root,
        )
        return

    parser = argparse.ArgumentParser(description="Run TUPL on one GZSDA episode")
    parser.add_argument("--data_root", type=str, default="../data/")
    parser.add_argument("--dataset", type=str, default="officehome", choices=list(DATASET_CONFIGS))
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--p", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--kernel_mode", type=str, default="paper", choices=["paper", "median"])
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--stage2_epochs", type=int, default=2000)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--variant", type=str, default="full", choices=["full", "phi_only", "psi_only"])
    parser.add_argument("--fusion", type=str, default="avg", choices=["phi", "psi", "avg", "subspace"])
    parser.add_argument("--metric", type=str, default="macro", choices=["macro", "sample"])
    parser.add_argument("--final_relu", action="store_true")
    parser.add_argument("--loss_reduction", type=str, default="sum", choices=["sum", "mean"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Sweep all trials, domain pairs, and comma-separated hyperparameter lists",
    )
    parser.add_argument("--trials", type=str, default=None, help="e.g. 0,1,2 or omit for all")
    parser.add_argument("--variants", type=str, default="full", help="e.g. full,phi_only,psi_only")
    parser.add_argument("--fusion_modes", type=str, default="avg")
    parser.add_argument("--alphas", type=str, default="1.0", help="e.g. 0.1,1.0,10.0")
    parser.add_argument("--kernel_modes", type=str, default="paper")
    parser.add_argument("--ps", type=str, default="1024")
    parser.add_argument("--ms", type=str, default="512")
    parser.add_argument("--stage2_epochs_list", type=str, default="2000")
    parser.add_argument("--stage2_lrs", type=str, default="1e-4")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    def _parse_int_list(text: str):
        return [int(x.strip()) for x in text.split(",") if x.strip()]

    def _parse_float_list(text: str):
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    def _parse_str_list(text: str):
        return [x.strip() for x in text.split(",") if x.strip()]

    if args.grid:
        trials = _parse_int_list(args.trials) if args.trials is not None else None
        results = run_experiment_grid(
            data_root=args.data_root,
            dataset=args.dataset,
            backbone=args.backbone,
            trials=trials,
            variants=tuple(_parse_str_list(args.variants)),
            fusion_modes=tuple(_parse_str_list(args.fusion_modes)),
            ps=tuple(_parse_int_list(args.ps)),
            alphas=tuple(_parse_float_list(args.alphas)),
            kernel_modes=tuple(_parse_str_list(args.kernel_modes)),
            ms=tuple(_parse_int_list(args.ms)),
            stage2_epochs_list=tuple(_parse_int_list(args.stage2_epochs_list)),
            stage2_lrs=tuple(_parse_float_list(args.stage2_lrs)),
            final_relu=args.final_relu,
            loss_reduction=args.loss_reduction,
            metric=args.metric,
            seed=args.seed,
            verbose=args.verbose,
            quiet=False,
            on_progress=lambda i, n, s, t, tr, var: print(
                f"[{i}/{n}] source={s} target={t} trial={tr} variant={var}"
            ),
        )
        export_results(results, args.output_dir, prefix=args.dataset)
        summary = summarize_results(results, group_by=("variant", "alpha"))
        print("\n=== summary (mean over runs) ===")
        for row in summary:
            print(row)
        return

    run_tupl(
        data_root=args.data_root,
        dataset=args.dataset,
        backbone=args.backbone,
        source=args.source,
        target=args.target,
        trial=args.trial,
        p=args.p,
        alpha=args.alpha,
        kernel_mode=args.kernel_mode,
        m=args.m,
        stage2_epochs=args.stage2_epochs,
        stage2_lr=args.stage2_lr,
        variant=args.variant,
        fusion=args.fusion,
        final_relu=args.final_relu,
        loss_reduction=args.loss_reduction,
        metric=args.metric,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
