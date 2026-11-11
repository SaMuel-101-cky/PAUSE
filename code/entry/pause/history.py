"""Cross-epoch history summary end positions."""
import torch


RANDOM_HISTORY_SEEDS = (2025, 2026, 2027, 2028, 2029)
RANDOM_HISTORY_END_IDX_DIST = "uniform[0,P-1]"
FULL_HISTORY_END_IDX_DIST = "all_P-1"


def history_end_idx_dist(history_mode):
    """Return the history sampling contract for zero-based ``end_idx``."""
    if history_mode == "random":
        return RANDOM_HISTORY_END_IDX_DIST
    if history_mode == "full":
        return FULL_HISTORY_END_IDX_DIST
    raise ValueError("history_mode must be 'full' or 'random'")


def _validate_shape(num_epochs, num_patches):
    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")
    if num_patches < 1:
        raise ValueError("num_patches must be >= 1")


def sample_training_end_idx(
    batch_size,
    num_epochs,
    num_patches,
    device,
):
    _validate_shape(num_epochs, num_patches)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return torch.randint(
        low=0,
        high=num_patches,
        size=(batch_size, num_epochs),
        device=device,
        dtype=torch.long,
    )


def build_random_history_tables(
    num_windows,
    num_epochs,
    num_patches,
    seeds=RANDOM_HISTORY_SEEDS,
):
    _validate_shape(num_epochs, num_patches)
    if num_windows < 1:
        raise ValueError("num_windows must be >= 1")
    tables = {}
    for seed in tuple(seeds):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        tables[int(seed)] = torch.randint(
            low=0,
            high=num_patches,
            size=(num_windows, num_epochs),
            generator=generator,
            device="cpu",
            dtype=torch.long,
        )
    return tables
