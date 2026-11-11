from dataclasses import dataclass

from config.pause import (
    MODEL_KWARGS,
    TRAIN_DEFAULTS,
    VARIANTS,
    PauseDataCfg,
    calib_dir as cfg_calib_dir,
    ckpt_dir as cfg_ckpt_dir,
    get_cfg,
)


DEFAULT_NUM_PATCHES = 30


@dataclass(frozen=True)
class GPDatasetProfile:
    """Declarative configuration for one PAUSE GP dataset/variant pairing."""

    dataset: str
    pause_cfg: PauseDataCfg
    s1_checkpoint_dir: str
    oof_logits_dir: str | None
    temperature_dir: str | None
    gp_checkpoint_dir: str
    gp_output_dir: str
    num_classes: int
    batch_size: int
    model_kwargs: dict
    num_history_draws: int
    model_variant: str | None
    calibration_history_mode: str | None
    history_seeds: tuple[int, ...]


def _variant_namespace(variant: str) -> str:
    return "pause" if variant == "paired" else f"pause_{variant}"


def _pause_profile(dataset: str, variant: str) -> GPDatasetProfile:
    cfg = get_cfg(dataset)
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown PAUSE variant {variant!r}; choices: {tuple(VARIANTS)}"
        )
    spec = VARIANTS[variant]
    namespace = _variant_namespace(variant)
    history_seeds = (
        tuple(range(2025, 2025 + 32))
        if spec.calib_history == "random"
        else ()
    )
    root = dataset.upper()
    num_classes = int(TRAIN_DEFAULTS["num_classes"])
    model_kwargs = {
        **dict(MODEL_KWARGS),
        "num_epochs": int(TRAIN_DEFAULTS["num_window_epochs"]),
        "num_patches": DEFAULT_NUM_PATCHES,
        "num_classes": num_classes,
    }
    return GPDatasetProfile(
        dataset=dataset,
        pause_cfg=cfg,
        s1_checkpoint_dir=cfg_ckpt_dir(cfg, variant),
        oof_logits_dir=f"{cfg_calib_dir(cfg, variant)}/oof_logits",
        temperature_dir=f"{cfg_calib_dir(cfg, variant)}/temperatures",
        gp_checkpoint_dir=f"checkpoint/gp/{namespace}/{root}",
        gp_output_dir=f"outputs/gp/{namespace}/{root}",
        num_classes=num_classes,
        batch_size=cfg.batch_size_windows,
        model_kwargs=model_kwargs,
        num_history_draws=len(history_seeds) if history_seeds else 1,
        model_variant=variant,
        calibration_history_mode=spec.calib_history,
        history_seeds=history_seeds,
    )


def get_gp_profile(
    dataset: str,
    variant: str,
) -> GPDatasetProfile:
    return _pause_profile(dataset, variant)
