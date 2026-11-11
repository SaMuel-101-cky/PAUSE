from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPConfig:
    """Configuration for fold-level gp orchestration."""

    penalty: float = 10.0
    penalty_alpha: float = 0.5
    cost_pool: list[float] = field(
        default_factory=lambda:
        [0.0005, 0.006667,
        0.02, 0.03333,
        0.06667, 0.13333,
        0.23333, 0.5]
    )


    gp_epochs: int = 20
    gp_batch_size: int = 8192
    num_inducing_points: int = 400
    gp_early_stopping: bool = True
    gp_validation_fraction: float = 0.05
    gp_min_epochs: int = 5
    gp_patience: int = 3
    gp_min_delta: float = 1e-4
    gp_learning_rate: float = 0.02
    gp_lr_reduce_factor: float = 0.25
    gp_max_lr_reductions: int = 1
    gp_early_stopping_seed: int = 2025
    is_llr_as_sufficient_statistic: bool = False

    num_classes: int = 5
    upper_horizon: Optional[int] = None

    seed: int = 2025
    log_dir: str = "outputs/logs"
    device: str = "cuda"
    posterior_device: str = "cpu"

    num_workers: int = 4
