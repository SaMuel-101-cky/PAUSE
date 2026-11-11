import math
from numbers import Integral

import torch


def _validate_num_classes(num_classes: int) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
        raise ValueError("num_classes must be a positive integer")
    num_classes = int(num_classes)
    if num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    return num_classes


def validate_penalty_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("penalty_alpha must be finite and non-negative")
    return alpha


def as_cost_matrix(cost_spec, num_classes: int, *, device, dtype) -> torch.Tensor:
    """Return a validated [true_class, predicted_class] cost matrix."""
    num_classes = _validate_num_classes(num_classes)
    cost = torch.as_tensor(cost_spec, dtype=dtype, device=device)
    if cost.ndim == 0:
        if not torch.isfinite(cost) or cost < 0:
            raise ValueError("scalar misclassification cost must be finite and non-negative")
        matrix = torch.full(
            (num_classes, num_classes),
            cost.item(),
            dtype=dtype,
            device=device,
        )
        matrix.fill_diagonal_(0)
        return matrix

    if cost.shape != (num_classes, num_classes):
        raise ValueError(
            "misclassification cost matrix must have shape "
            f"({num_classes}, {num_classes})"
        )
    if not torch.isfinite(cost).all() or torch.any(cost < 0):
        raise ValueError("misclassification cost matrix must be finite and non-negative")
    if not torch.all(cost.diagonal() == 0):
        raise ValueError("misclassification cost matrix diagonal must be zero")
    return cost


def build_class_weighted_cost_matrix(
    labels,
    *,
    num_classes: int,
    penalty: float,
    alpha: float,
) -> tuple[torch.Tensor, dict]:
    """Build a source-fold cost matrix and JSON-safe reproducibility metadata."""
    num_classes = _validate_num_classes(num_classes)
    alpha = validate_penalty_alpha(alpha)
    penalty = float(penalty)
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and non-negative")

    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a non-empty one-dimensional tensor")
    if torch.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("labels must be in [0, num_classes)")

    counts = torch.bincount(labels, minlength=num_classes)
    if alpha > 0 and torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"cannot weight missing training classes: {missing}")

    counts_float = counts.to(torch.float32)
    sample_count = float(labels.numel())
    if alpha == 0:
        raw_weights = torch.ones(num_classes, dtype=torch.float32)
    else:
        raw_weights = (sample_count / (num_classes * counts_float)).pow(alpha)
    empirical_prior = counts_float / sample_count
    normalized_weights = raw_weights / torch.dot(empirical_prior, raw_weights)
    matrix = torch.outer(
        penalty * normalized_weights,
        torch.ones(num_classes, dtype=torch.float32),
    )
    matrix.fill_diagonal_(0)

    metadata = {
        "penalty": penalty,
        "penalty_alpha": alpha,
        "class_counts": counts.tolist(),
        "raw_weights": raw_weights.tolist(),
        "normalized_weights": normalized_weights.tolist(),
        "cost_matrix": matrix.tolist(),
    }
    return matrix, metadata


def expected_misclassification_costs(
    posteriors: torch.Tensor,
    cost_spec,
) -> torch.Tensor:
    """Return the expected cost of every prediction for posterior prefixes."""
    posteriors = torch.as_tensor(posteriors)
    if posteriors.ndim != 3:
        raise ValueError(f"posteriors must be [B,T,K], got {posteriors.shape}")
    if not posteriors.is_floating_point():
        posteriors = posteriors.float()
    matrix = as_cost_matrix(
        cost_spec,
        posteriors.shape[-1],
        device=posteriors.device,
        dtype=posteriors.dtype,
    )
    return torch.matmul(posteriors, matrix)
