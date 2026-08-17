# changeDomain

Research repo for **Generalized Zero-Shot Domain Adaptation (GZSDA)** on images and a custom **ActionStyle** motion dataset.

GZSDA: source is labeled for **all** classes; target has labels only for **seen** classes. Test is target **seen + unseen**. Metrics: per-class **Acc_s**, **Acc_u**, harmonic **H-mean**. No class semantics/attributes (unlike GZSL).

Active work lives in `notebooks/effective-gzsda/gzsda/`. Pose/data-prep is an earlier pipeline that produced ActionStyle features.

## Layout

```
notebooks/effective-gzsda/gzsda/   # main experiments (cwd for notebooks)
  expriment-*.ipynb                # entry points (typo in filename is intentional)
  src/                             # methods: utils, tupl, our_tupl, vista_gzsda, models
  data/                            # .mat features + splits (gitignored)
  result/{json,csv}/               # saved Acc_s / Acc_u / H per domain pair
  paper/                           # TUPL + VisTA paper notes
  VisTA/                           # vendored official VisTA (CI-UDA, RGB+CLIP) — do not treat as GZSDA code
data-preparation/                  # SMPLEST-X pose extract + Blender retarget
notebooks/pose/                    # mocap → SMPL → MotionCLIP embeddings
resources/models/smplh/            # SMPL-H assets (gitignored binaries)
src/                               # empty placeholder — real code is under gzsda/src
```

## Run experiments from `notebooks/effective-gzsda/gzsda`

Notebooks import `from src....`. Do not run them from the repo root.

| Notebook | Domains | Classes | Features | Dim | Trials | `NUM_LABELS` |
|---|---|---|---|---|---|---|
| `expriment-office31.ipynb` | A, D, W | 31 | ResNet-50 `.mat` | 2048 | 5 | 31 |
| `expriment-officeHome.ipynb` | Art, Clipart, Product, RealWorld | 65 | ResNet-50 `.mat` | 2048 | 5 | 65 |
| `expriment-xray.ipynb` | regu, xray | 20 | ResNet-101 `.mat` | 2048 | 5 | 20 |
| `expriment-actionStyle.ipynb` | 7 styles (angry…strutting) | 4 actions | MotionCLIP/CLIP `.mat` | 512 | 6 | 5 |

ActionStyle classes: `punch, jump, kick, walk` (ids 0–3). `NUM_LABELS = 5` is leftover from when `run` existed — keep it unless you also change classifier heads.

ActionStyle VAE args in the notebook: encoder/decoder `[512, 512]`. Default `get_args` is `[2048, 512]` / `[512, 2048]`.

Dataset configs and paper H-targets: `src/tupl.py` → `DATASET_CONFIGS`.

### Methods (result keys)

| Key | What |
|---|---|
| `base` | Linear classifier on real source + labeled target seen |
| `CCVAE` | Cross-domain VAE (`get_trained_VAE`) then classifier on real + generated |
| `our0` | Same VAE, different classifier mix policy (`main_m0` in notebook) |
| `our_GRE` | VAE + domain classifier on latent z (`get_trained_VAE_with_domain_classifier`) |
| `TUPL` | Paper prototype method (`src/tupl.py`) |
| `our_TUPL_<policy>` | VAE-generated features fed into TUPL (`src/our_tupl.py`) |
| `VisTA` | Feature-space VisTA (`src/vista_gzsda.py`) — ActionStyle only so far |

`our_TUPL` policies: `real_plus_src2tgt`, `real_plus_src2tgt_unseen`, `interp_src2tgt`.

Official `VisTA/` is CI-UDA on images (Dassl + CLIP ViT + Grad-CAM). `src/vista_gzsda.py` is the GZSDA port: frozen CLIP **text** encoder, no image encoder, cosine retrieval instead of Grad-CAM VAC.

### Results protocol

1. Load `./result/json/<name>.json` if present; skip a method if its key already exists.
2. After a method finishes, write JSON immediately.
3. Export `./result/csv/<name>.csv` from the JSON.

Do **not** clear `result.pop(...)` unless the user asks to rerun. Full grids are GPU-heavy (all domain pairs × trials).

`run_all_senario(main, DOMAIN_SET, input_dim=..., num_trial=...)` in `src/utils.py` loops every `source != target` pair. Return value of `main` must be `(acc_per_class, overall, acc_seen, acc_unseen)` as **fractions**.

Reports print percent: `mean ± SEM`. H-mean is computed from Acc_s/Acc_u.

### Data files (not in git)

`data/.gitignore` excludes `Office31/`, `OfficeHome/`, `XrayBaggage20/`, `ActionStyleDataset/`, `*.npz`.

Per-domain `.mat`: `labels` `(1, N)` and a feature key (e.g. `resnet50_features` or `clip_features`) shaped `(N, D, 1, 1)`. Split file: `targetDomain_splitFlag` / `targetDomain_unseenClass` nested by trial then domain. Flags: `1` train, `2` test, `0` unused. Unseen target train samples are paired with `yT = -1`.

Rebuild ActionStyle mats with `data/create-dataset.ipynb` from `inpDataset.npz`.

GZSDA baseline data: https://github.com/hellowangqian/gzsda

## Pose / ActionStyle pipeline (upstream)

Use only when changing how ActionStyle features are built.

1. `data-preparation/blender/` — Mixamo characters + BVH mocap → rendered videos (`4.create_dataset.ipynb` calls Blender).
2. `data-preparation/1.PoseExtractor_SMPLEST-X/` — video → SMPL (vendored).
3. `notebooks/pose/1.change_format` → `2.check_validity` → `3.mocap_to_smpl` → `4.motion_clip`.
4. MotionCLIP encode (`notebooks/pose/4.motion_clip/4.create_dataset.ipynb`; env `motionclip_py310_v2`) → npz → GZSDA `.mat`.

Vendored: `MotionCLIP/`, `1.PoseExtractor_SMPLEST-X/`. Prefer editing notebooks and `gzsda/src`, not upstream forks.

## Coding conventions

- Prefer extending `gzsda/src/*.py` and calling from notebooks. Keep `main_*` wrappers in the notebook.
- Do not rename `expriment-*`, `get_datesets_and_loaders`, or `run_all_senario` unless the user asks (call sites everywhere).
- Do not reformat or “clean up” vendored `VisTA/`, `MotionCLIP/`, SMPLEST-X.
- Features are L2-normalized in `BaseTwoModalDataset` / `TUPLDataset`.
- Default conda env for GZSDA notebooks: `asad` (PyTorch). MotionCLIP uses `motionclip_py310_v2`.
- SciPy may warn about NumPy 2.x; ignore unless imports fail.
- Commit style: `[notebook][effective-gzsda][gzsda] <what>` or `[pipeline][components] <what>`.
- Do not commit `.mat`, `.npz`, checkpoints, SMPL weights, or rendered videos.

## Quick pointers

- TUPL implementation + Office-31 paper H-targets: `src/tupl.py`
- VAE / classifier / dataset loader: `src/utils.py`, `src/models.py`
- VAE→TUPL: `src/our_tupl.py`
- Feature-space VisTA: `src/vista_gzsda.py` (`python -m src.vista_gzsda` from `gzsda/`)
- Dump notebook/src text: `extractor.sh` (paths inside are stale)
