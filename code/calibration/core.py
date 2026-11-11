"""PAUSE cross-fold per-k temperature calibration core — posterior and reliability statistics."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, recall_score


@dataclass(frozen=True)
class FoldLogits:
    """Per-k logits and ground-truth labels for one subject in a cross-validation fold."""

    subject_id: int
    logits: torch.Tensor
    labels: torch.Tensor
    log_prior: torch.Tensor

    def __post_init__(self):
        logits = torch.as_tensor(self.logits, device="cpu")
        labels = torch.as_tensor(self.labels, device="cpu")
        log_prior = torch.as_tensor(self.log_prior, device="cpu")

        if logits.ndim != 3:
            raise ValueError(
                f"logits must be [N,T,K], got {tuple(logits.shape)}"
            )
        if logits.is_complex():
            raise ValueError("logits must be real-valued")
        if logits.shape[0] < 1 or logits.shape[1] < 1 or logits.shape[2] < 1:
            raise ValueError("logits dimensions N, T, and K must be positive")
        if labels.shape != (logits.shape[0],):
            raise ValueError(
                f"labels must be [N] with N={logits.shape[0]}, "
                f"got {tuple(labels.shape)}"
            )
        if log_prior.shape != (logits.shape[2],):
            raise ValueError(
                f"log_prior must be [K] with K={logits.shape[2]}, "
                f"got {tuple(log_prior.shape)}"
            )

        _require_finite("logits", logits)
        _require_finite("labels", labels)
        _require_finite("log_prior", log_prior)
        if (
            labels.dtype == torch.bool
            or labels.is_floating_point()
            or labels.is_complex()
        ):
            raise ValueError("labels must contain integers")
        if log_prior.is_complex():
            raise ValueError("log_prior must be real-valued")

        logits = logits.detach().to(dtype=torch.float64)
        labels = labels.detach().to(dtype=torch.long)
        log_prior = log_prior.detach().to(dtype=torch.float64)
        if torch.any((labels < 0) | (labels >= logits.shape[2])):
            raise ValueError("labels must be in [0, K)")

        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "log_prior", log_prior)


@dataclass(frozen=True)
class TemperatureFit:
    """Temperature fit result for a single k-index.

    nll_before is the NLL at T=1 (uncalibrated baseline).
    """

    temperature: float
    nll_before: float
    nll_after: float
    status: str
    fallback_used: bool


def _require_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def subject_equal_nll(log_temperature, sources, k_index):
    """First average over epochs per subject, then equal-weight average over subjects."""
    if sources is None:
        raise ValueError("sources must not be empty")
    sources = tuple(sources)
    if not sources:
        raise ValueError("sources must not be empty")
    if isinstance(k_index, bool) or not isinstance(k_index, int):
        raise ValueError("k_index must be an integer")
    if k_index < 0:
        raise ValueError("k_index is out of range")

    log_temperature = torch.as_tensor(
        log_temperature,
        dtype=torch.float64,
        device="cpu",
    )
    if log_temperature.ndim != 0:
        raise ValueError("log_temperature must be scalar")
    temperature = torch.exp(log_temperature)
    losses = []
    for source in sources:
        if not isinstance(source, FoldLogits):
            raise TypeError("sources must contain FoldLogits values")
        if k_index >= source.logits.shape[1]:
            raise ValueError("k_index is out of range")
        scores = (
            source.logits[:, k_index, :] / temperature
            + source.log_prior
        )
        losses.append(
            F.cross_entropy(scores, source.labels, reduction="mean")
        )
    return torch.stack(losses).mean()


def _nll_at_log_temperature(log_temperature, sources, k_index):
    with torch.no_grad():
        loss = subject_equal_nll(
            torch.tensor(log_temperature, dtype=torch.float64),
            sources,
            k_index,
        )
    return float(loss.item())


def _bounded_temperature_fit(
    sources,
    k_index,
    min_temperature=0.05,
    max_temperature=20.0,
):
    """Search for log(T) within a given temperature interval using gradient-free scalar optimization."""

    def objective(log_temperature):
        value = _nll_at_log_temperature(
            log_temperature,
            sources,
            k_index,
        )
        return value if math.isfinite(value) else math.inf

    return minimize_scalar(
        objective,
        bounds=(math.log(min_temperature), math.log(max_temperature)),
        method="bounded",
        options={"xatol": 1e-8},
    )


def _valid_temperature_candidate(
    temperature,
    nll,
    baseline,
    min_temperature,
    max_temperature,
    tolerance,
):
    return (
        math.isfinite(temperature)
        and math.isfinite(nll)
        and min_temperature <= temperature <= max_temperature
        and nll <= baseline + tolerance
    )


def fit_temperature_for_k(
    sources,
    k_index,
    min_temperature=0.05,
    max_temperature=20.0,
    max_iter=50,
    tolerance=1e-10,
):
    """Fit per-k temperature via LBFGS; fall back to bounded scalar optimization on failure."""
    if (
        isinstance(min_temperature, bool)
        or isinstance(max_temperature, bool)
        or not isinstance(min_temperature, (int, float))
        or not isinstance(max_temperature, (int, float))
        or not math.isfinite(min_temperature)
        or not math.isfinite(max_temperature)
        or min_temperature <= 0
        or min_temperature >= max_temperature
    ):
        raise ValueError(
            "temperature bounds must be finite, positive, and ordered"
        )
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise ValueError("max_iter must be a positive integer")
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("tolerance must be finite and non-negative")

    sources = tuple(sources) if sources is not None else ()
    baseline = _nll_at_log_temperature(0.0, sources, k_index)
    if not math.isfinite(baseline):
        raise ValueError("baseline NLL is not finite")

    log_temperature = torch.tensor(
        0.0,
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = subject_equal_nll(log_temperature, sources, k_index)
        if not torch.isfinite(loss):
            raise RuntimeError("LBFGS produced a non-finite NLL")
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        candidate_log_temperature = float(log_temperature.detach().item())
        candidate_temperature = math.exp(candidate_log_temperature)
        candidate_nll = _nll_at_log_temperature(
            candidate_log_temperature,
            sources,
            k_index,
        )
        if _valid_temperature_candidate(
            candidate_temperature,
            candidate_nll,
            baseline,
            min_temperature,
            max_temperature,
            tolerance,
        ):
            return TemperatureFit(
                temperature=candidate_temperature,
                nll_before=baseline,
                nll_after=candidate_nll,
                status="lbfgs",
                fallback_used=False,
            )
    except (RuntimeError, OverflowError, ValueError):
        pass

    try:
        bounded = _bounded_temperature_fit(
            sources,
            k_index,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
        )
        bounded_log_temperature = float(bounded.x)
        bounded_temperature = math.exp(bounded_log_temperature)
        bounded_nll = _nll_at_log_temperature(
            bounded_log_temperature,
            sources,
            k_index,
        )
        if (
            bool(bounded.success)
            and _valid_temperature_candidate(
                bounded_temperature,
                bounded_nll,
                baseline,
                min_temperature,
                max_temperature,
                tolerance,
            )
            and bounded_nll < baseline
        ):
            return TemperatureFit(
                temperature=bounded_temperature,
                nll_before=baseline,
                nll_after=bounded_nll,
                status="bounded",
                fallback_used=True,
            )
    except (RuntimeError, OverflowError, ValueError):
        pass

    return TemperatureFit(
        temperature=1.0,
        nll_before=baseline,
        nll_after=baseline,
        status="identity",
        fallback_used=True,
    )


def calibrated_log_probs(logits, log_prior, temperatures):
    """Compute log-probabilities of ``softmax(logits / temperature + log_prior)``."""
    logits = torch.as_tensor(logits)
    log_prior = torch.as_tensor(
        log_prior,
        dtype=logits.dtype,
        device=logits.device,
    )
    temperatures = torch.as_tensor(
        temperatures,
        dtype=logits.dtype,
        device=logits.device,
    )

    if logits.ndim != 3:
        raise ValueError(f"logits must be [N,T,K], got {tuple(logits.shape)}")
    if log_prior.shape != (logits.shape[-1],):
        raise ValueError(
            f"log_prior must be [{logits.shape[-1]}], "
            f"got {tuple(log_prior.shape)}"
        )
    if temperatures.shape != (logits.shape[1],):
        raise ValueError(
            f"temperatures must be [{logits.shape[1]}], "
            f"got {tuple(temperatures.shape)}"
        )

    _require_finite("logits", logits)
    _require_finite("log_prior", log_prior)
    _require_finite("temperatures", temperatures)
    if not torch.all(temperatures > 0):
        raise ValueError("temperatures must be positive")

    scores = (
        logits / temperatures.view(1, -1, 1)
        + log_prior.view(1, 1, -1)
    )
    _require_finite("scores", scores)
    log_probs = F.log_softmax(scores, dim=-1)
    _require_finite("log_probs", log_probs)
    return log_probs


def reliability_bins(confidence, correct, n_bins=15):
    """Fixed equal-width binning; empty bins get zero statistics with count=0."""
    confidence = torch.as_tensor(confidence, dtype=torch.float64)
    correct = torch.as_tensor(correct, dtype=torch.float64)

    if confidence.ndim != 1 or correct.ndim != 1:
        raise ValueError("confidence and correct must be one-dimensional")
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same shape")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    _require_finite("confidence", confidence)
    _require_finite("correct", correct)
    if torch.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be in [0, 1]")
    if torch.any((correct != 0) & (correct != 1)):
        raise ValueError("correct must contain only 0 or 1")

    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    bin_confidence = np.zeros(n_bins, dtype=np.float64)
    accuracy = np.zeros(n_bins, dtype=np.float64)

    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (
                (confidence >= edges[index])
                & (confidence <= edges[index + 1])
            )
        else:
            mask = (
                (confidence >= edges[index])
                & (confidence < edges[index + 1])
            )

        count[index] = int(mask.sum().item())
        if count[index]:
            bin_confidence[index] = float(confidence[mask].mean().item())
            accuracy[index] = float(correct[mask].mean().item())

    return {
        "lower": edges[:-1].numpy(),
        "upper": edges[1:].numpy(),
        "count": count,
        "confidence": bin_confidence,
        "accuracy": accuracy,
    }


def _ece_from_stats(stats):
    total = int(stats["count"].sum())
    if total == 0:
        return 0.0

    weights = stats["count"].astype(np.float64) / total
    return float(
        np.sum(
            weights
            * np.abs(stats["accuracy"] - stats["confidence"])
        )
    )


def confidence_ece(probs, labels, n_bins=15):
    """Compute top-confidence ECE for multiclass predictions."""
    probs = torch.as_tensor(probs, dtype=torch.float64)
    labels = torch.as_tensor(labels)

    if probs.ndim != 2:
        raise ValueError("probs must be [N,K]")
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if probs.shape[1] < 1:
        raise ValueError("probs must contain at least one class")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs must be [N,K] and labels must be [N]")
    if (
        labels.dtype == torch.bool
        or labels.is_floating_point()
        or labels.is_complex()
    ):
        raise ValueError("labels must contain integers")

    _require_finite("probs", probs)
    if torch.any((probs < 0) | (probs > 1)):
        raise ValueError("probs must be in [0, 1]")
    row_sums = probs.sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=1e-5,
        atol=1e-8,
    ):
        raise ValueError("each probs row must sum to 1")
    if torch.any((labels < 0) | (labels >= probs.shape[1])):
        raise ValueError("labels must be in [0, K)")

    confidence, prediction = probs.max(dim=-1)
    correct = prediction.eq(labels)
    return _ece_from_stats(
        reliability_bins(confidence, correct, n_bins)
    )


def binary_ece(probability, target, n_bins=15):
    """Compute ECE of a one-vs-rest probability against a binary target."""
    probability = torch.as_tensor(
        probability,
        dtype=torch.float64,
    )
    target = torch.as_tensor(target, dtype=torch.float64)
    if probability.ndim != 1 or target.ndim != 1:
        raise ValueError("probability and target must be one-dimensional")
    return _ece_from_stats(
        reliability_bins(probability, target, n_bins)
    )


@dataclass(frozen=True)
class TargetTemperatureResult:
    """Temperature matrix fitted per-k on all folds except the target subject."""

    target_subject: int
    source_subjects: np.ndarray
    temperature: np.ndarray
    nll_fit_before: np.ndarray
    nll_fit_after: np.ndarray
    optimizer_status: np.ndarray
    fallback_used: np.ndarray


def fit_target_temperatures(folds, target_subject):
    """Fit per-k temperatures on all folds excluding the target subject."""
    ids = [fold.subject_id for fold in folds]
    if len(ids) != len(set(ids)):
        raise ValueError("subject ids must be unique")
    if target_subject not in ids:
        raise ValueError("target_subject is missing from folds")
    steps = {fold.logits.shape[1] for fold in folds}
    classes = {fold.logits.shape[2] for fold in folds}
    if len(steps) != 1 or len(classes) != 1:
        raise ValueError("all folds must share T and K")
    sources = sorted(
        (fold for fold in folds if fold.subject_id != target_subject),
        key=lambda fold: fold.subject_id,
    )
    if len(sources) != len(folds) - 1:
        raise AssertionError("target exclusion failed")

    fits = [
        fit_temperature_for_k(sources, k_index)
        for k_index in range(next(iter(steps)))
    ]
    return TargetTemperatureResult(
        target_subject=target_subject,
        source_subjects=np.asarray(
            [source.subject_id for source in sources], dtype=np.int64
        ),
        temperature=np.asarray([fit.temperature for fit in fits], dtype=np.float64),
        nll_fit_before=np.asarray([fit.nll_before for fit in fits], dtype=np.float64),
        nll_fit_after=np.asarray([fit.nll_after for fit in fits], dtype=np.float64),
        optimizer_status=np.asarray([fit.status for fit in fits], dtype="<U16"),
        fallback_used=np.asarray([fit.fallback_used for fit in fits], dtype=np.bool_),
    )


def _per_k_nll(log_probs, labels):
    rows = torch.arange(labels.shape[0])
    return np.asarray(
        [
            float((-log_probs[rows, k, labels]).mean().item())
            for k in range(log_probs.shape[1])
        ],
        dtype=np.float64,
    )


def evaluate_fold_calibration(fold, temperatures, *, n_bins=15, n1_index=1):
    """Evaluate a single target subject; temperatures must be fitted from other folds."""
    temperatures = np.asarray(temperatures, dtype=np.float64)
    steps = fold.logits.shape[1]
    classes = fold.logits.shape[2]
    if temperatures.shape != (steps,):
        raise ValueError(f"temperatures must be [{steps}]")
    if not 0 <= n1_index < classes:
        raise ValueError("n1_index out of range")

    before = calibrated_log_probs(
        fold.logits, fold.log_prior, torch.ones(steps, dtype=torch.float64)
    )
    after = calibrated_log_probs(fold.logits, fold.log_prior, temperatures)
    before_probs = before.exp()
    after_probs = after.exp()
    labels_np = fold.labels.numpy()
    all_labels = list(range(classes))

    result = {
        "ece_before": np.zeros(steps),
        "ece_after": np.zeros(steps),
        "nll_before": _per_k_nll(before, fold.labels),
        "nll_after": _per_k_nll(after, fold.labels),
        "n1_ece_before": np.zeros(steps),
        "n1_ece_after": np.zeros(steps),
        "map_acc_before": np.zeros(steps),
        "map_acc_after": np.zeros(steps),
        "map_mf1_before": np.zeros(steps),
        "map_mf1_after": np.zeros(steps),
        "map_recall_before": np.zeros((steps, classes)),
        "map_recall_after": np.zeros((steps, classes)),
        "ml_acc": np.zeros(steps),
        "ml_mf1": np.zeros(steps),
    }
    ml_before = fold.logits.argmax(-1)
    ml_after = (
        fold.logits
        / torch.as_tensor(temperatures, dtype=torch.float64).view(1, -1, 1)
    ).argmax(-1)
    result["ml_predictions_equal"] = np.asarray(torch.equal(ml_before, ml_after))

    is_n1 = fold.labels.eq(n1_index)
    for k in range(steps):
        result["ece_before"][k] = confidence_ece(
            before_probs[:, k], fold.labels, n_bins
        )
        result["ece_after"][k] = confidence_ece(
            after_probs[:, k], fold.labels, n_bins
        )
        result["n1_ece_before"][k] = binary_ece(
            before_probs[:, k, n1_index], is_n1, n_bins
        )
        result["n1_ece_after"][k] = binary_ece(
            after_probs[:, k, n1_index], is_n1, n_bins
        )
        before_pred = before_probs[:, k].argmax(-1).numpy()
        after_pred = after_probs[:, k].argmax(-1).numpy()
        ml_pred = ml_before[:, k].numpy()
        for suffix, prediction in (
            ("before", before_pred),
            ("after", after_pred),
        ):
            result[f"map_acc_{suffix}"][k] = accuracy_score(labels_np, prediction)
            result[f"map_mf1_{suffix}"][k] = f1_score(
                labels_np,
                prediction,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
            result[f"map_recall_{suffix}"][k] = recall_score(
                labels_np,
                prediction,
                labels=all_labels,
                average=None,
                zero_division=0,
            )
        result["ml_acc"][k] = accuracy_score(labels_np, ml_pred)
        result["ml_mf1"][k] = f1_score(
            labels_np,
            ml_pred,
            labels=all_labels,
            average="macro",
            zero_division=0,
        )
    if not bool(result["ml_predictions_equal"]):
        raise AssertionError("positive temperature changed ML predictions")
    return result
