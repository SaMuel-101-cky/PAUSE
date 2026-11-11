"""Cross-epoch PAUSE unified configuration registry for four datasets.

Shared MODEL_KWARGS/TRAIN_DEFAULTS are declared once; each dataset only lists
differences in REGISTRY. Entry points look up PauseDataCfg by --dataset name.
"""
from dataclasses import dataclass


# Default model parameters
MODEL_KWARGS = {
    "embed_dim": 128,
    "num_heads": 8,
    "depth": 7,
    "mlp_ratio": 4.0,
    "dropout": 0.2,
    "in_dim": 128,
    "rope_mode": "2d",
    "rope_base": 30,
    "causal_patch_stem": True,
    "patch_stem_left_context": 4,
    "activation": "swiglu",
}


@dataclass(frozen=True)
class VariantSpec:
    """Training variant definition: single source of truth, shared by train/calibrate."""
    train_history: str        # training history builder selection
    cross_epoch_memory: bool  # model-side cross-epoch memory toggle
    calib_history: str        # unmoderated calibration history convention
    log_label: str            # display name for logging


VARIANTS = {
    "paired": VariantSpec("paired", True, "random", "V-Paired"),
    "full": VariantSpec("full", True, "full", "V-Full"),
    "random": VariantSpec("random", True, "random", "V-Random"),
    "nohist": VariantSpec("full", False, "full", "V-NoHist"),
}


# Default training parameters
TRAIN_DEFAULTS = {
    "num_train_epochs": 100,
    "num_window_epochs": 32,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "min_lr": 1e-6,
    "grad_clip": 1.0,
    "seed": 2025,
    "num_classes": 5,
    "early_stop_patience": 40,   # stop early after 40 consecutive epochs without a new best
}


@dataclass(frozen=True)
class PauseDataCfg:
    name: str
    data_dir: str
    fold_scheme: str          # "loso" | "grouped_kfold" | "file_kfold"
    num_folds: int            # ignored for LOSO (=number of subjects); used for k-fold
    batch_size_windows: int   # DataLoader batch size, unit=windows (1 window = num_window_epochs epochs)
    ckpt_name_fmt: str        # checkpoint filename template; DREAMS uses subject{fold_id}, others use fold{fold_id}


REGISTRY = {
    "dreams": PauseDataCfg("dreams", "dataset/DREAMS", "loso", 20, 16, "subject{fold_id}.pth"),
    "edf20": PauseDataCfg("edf20", "dataset/EDF20", "grouped_kfold", 20, 16, "fold{fold_id}.pth"),
    "edf78": PauseDataCfg("edf78", "dataset/EDF78", "grouped_kfold", 20, 32, "fold{fold_id}.pth"),
    "shhs": PauseDataCfg("shhs", "dataset/SHHS", "file_kfold", 20, 32, "fold{fold_id}.pth"),
}


def get_cfg(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset '{name}', choices: {sorted(REGISTRY)}")
    return REGISTRY[name]


def _namespace(variant):
    return "pause" if variant == "paired" else f"pause_{variant}"


def output_dir(cfg, variant="paired"):
    return f"outputs/{_namespace(variant)}/{cfg.name.upper()}"


def ckpt_dir(cfg, variant="paired"):
    return f"checkpoint/{_namespace(variant)}/{cfg.name.upper()}"


def calib_dir(cfg, variant="paired"):
    return f"outputs/calibration/{_namespace(variant)}/{cfg.name.upper()}"


def cache_stem_fmt(cfg):
    """Calibration cache / temperature / metrics filename stem (without .pth suffix)."""
    return cfg.ckpt_name_fmt.removesuffix(".pth")


def artifact_fold_id(cfg, fold_id):
    """Return the zero-based fold id used by ECTS baseline artifacts.

    PAUSE checkpoint/cache names are expressed directly by ``ckpt_name_fmt``;
    the ECTS baseline directory convention is always ``fold0..fold19``.
    """
    fold_id = int(fold_id)
    return fold_id - 1 if cfg.name == "dreams" else fold_id
