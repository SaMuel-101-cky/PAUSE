# PAUSE

Prefix-Adaptive and Utility-Aware Stopping for EEG-based resource-efficient sleep staging.

This repository contains the PAUSE implementation used in the paper. The code includes stage-1 training, out-of-fold calibration, and stage-2 GP stopping. Baseline implementations, raw datasets, trained checkpoints, and saved experiment results are not included.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `entry/pause/` | PAUSE stage-1 training, calibration, evaluation, and export entry points. |
| `entry/gp/` | PAUSE stage-2 Gaussian-process stopping and backward-induction orchestration. |
| `models/pause/` | Cross-epoch PAUSE classifier modules. |
| `models/gp/` | GP stopping model, cost functions, decision utilities, metrics, and artifact IO. |
| `training/` | PAUSE training loops and shared training utilities. |
| `config/pause.py` | Dataset registry, PAUSE variants, model defaults, and artifact path helpers. |
| `config/gp.py` | Fold-level GP orchestration defaults. |


## Environment

Run all commands from the `code/` directory with module syntax:

```bash
python -m <module> <args>
```

The supplied `requirements.txt` specifies a Python 3.11, Linux/CUDA 12.8 environment. Create an isolated environment and install it with `python -m pip install -r requirements.txt`. The pinned CUDA PyTorch wheel is not a macOS/CPU installation recipe; for those platforms, install an appropriate PyTorch wheel and the remaining dependencies separately.


## Data

Datasets are resolved through `config.pause.get_cfg(<dataset>)`. Registered dataset names are:

```text
dreams, edf20, edf78, shhs
```

The code expects the processed fold-compatible data files under the dataset roots declared in `config/pause.py`. All PAUSE variants use the same fold enumeration protocol from `dataset.folds`, so reported fold IDs are aligned across methods.

## PAUSE Pipeline

### 1. Train Stage-1 PAUSE

```bash
python -m entry.pause.train --dataset dreams --variant paired --device cuda
```

Useful options:

```bash
python -m entry.pause.train --dataset dreams --variant paired --smoke --device cpu
python -m entry.pause.train --dataset shhs --variant paired --workers 4 --device cuda
```

Default artifacts:

```text
checkpoint/pause/<DATASET>/
outputs/pause/<DATASET>/
outputs/logs/
```

### 2. Calibrate Prefix Posteriors

```bash
python -m entry.pause.calibrate --dataset dreams --variant paired --stage all --device cuda
```

Calibration writes out-of-fold logits and per-prefix temperatures:

```text
outputs/calibration/pause/<DATASET>/oof_logits/
outputs/calibration/pause/<DATASET>/temperatures/
```

The calibration entry point is variant-aware. Ablation variants use the namespaces defined in `config.pause.VARIANTS`.

### 3. Run GP Stopping

```bash
python -m entry.gp.gp --dataset dreams --variant paired --device cuda --posterior-device cpu
```

Single-fold or single-cost forward runs can be launched with:

```bash
python -m entry.gp.gp --dataset dreams --variant paired --fold_ids 0 --costs 0.001 --device cuda --posterior-device cpu
```

Default artifacts:

```text
checkpoint/gp/pause/<DATASET>/
outputs/gp/pause/<DATASET>/
```

The GP profile is built from `config.pause`, including checkpoint directories, calibration directories, PAUSE model kwargs, and ablation variant metadata.

## Ablations

PAUSE variants are declared in `config.pause.VARIANTS` and can be passed consistently to training, calibration, and GP:

```bash
python -m entry.pause.train --dataset dreams --variant nohist --device cuda
python -m entry.pause.calibrate --dataset dreams --variant nohist --stage all --device cuda
python -m entry.gp.gp --dataset dreams --variant nohist --device cuda --posterior-device cpu
```

Common variants include the full paired setting and historical ablations such as full-history or no-history calibration, depending on the registry.


## Output Conventions

| Component | Checkpoints | Outputs |
| --- | --- | --- |
| PAUSE stage-1 | `checkpoint/pause/<DATASET>/` | `outputs/pause/<DATASET>/` |
| PAUSE calibration | N/A | `outputs/calibration/pause/<DATASET>/` |
| PAUSE GP | `checkpoint/gp/pause/<DATASET>/` | `outputs/gp/pause/<DATASET>/` |

Most summary artifacts are written as CSV, JSON, or compressed NPZ files so they can be consumed directly by plotting scripts.

## Notes

- Training `--smoke` runs one epoch on the first fold and still requires processed data; it is not a data-free import check.
- Use `--fold_ids` and `--costs` for focused GP forward runs used in appendix plots.


## Processed data contract

Obtain datasets under their respective access terms and follow the preprocessing description in Section 2 of the supplementary PDF. This package does not redistribute participant data.

Each input MAT file must contain `PSD100` with shape `[N, 30, 128]` and a `label` array with `N` entries, encoded as 0=W, 1=N1, 2=N2, 3=N3, 4=REM. Files are expected directly in these directories:

| Dataset | Directory | Filename convention |
| --- | --- | --- |
| DREAMS-SUB | `dataset/DREAMS/` | `subject1.mat` through `subject20.mat` |
| Sleep-EDF-20 | `dataset/EDF20/` | `SC4*.mat`; subject identity is parsed from characters 4–5 |
| Sleep-EDF-78 | `dataset/EDF78/` | `SC4*.mat`; subject identity is parsed from characters 4–5 |
| SHHS | `dataset/SHHS/` | `shhs*-<numeric-subject-id>.mat` (for example, `shhs1-200001.mat`) |

`dataset.folds` defines the subject-level evaluation splits. The legacy `dataset/prepare.py` creates an epoch-level split and is not the PAUSE evaluation protocol; do not use it to prepare PAUSE evaluation folds. `dataset/make_dataset.py` handles an already extracted SHHS MAT representation, not raw EDF ingestion. The package does not include an end-to-end raw-data conversion pipeline for every dataset.

DREAMS stage-1 filenames use subject IDs 1–20. The GP `--fold_ids` argument uses canonical IDs 0–19 for all datasets, so `--fold_ids 0` corresponds to DREAMS subject 1. The example above is intentional.

Calibration requires stage-1 checkpoints and training-side out-of-fold predictions. Run training, calibration, and GP stopping in the stated order. Held-out evaluation subjects must not be used to fit temperatures.
