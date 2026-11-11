"""Cross-epoch classifier full/random-history evaluation."""
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score

from training.core import (
    evaluate_per_k,
    ml_map_predictions,
    summary_at_k,
)


@dataclass(frozen=True)
class CollectedOutputs:
    logits: torch.Tensor
    labels: torch.Tensor
    end_idx: torch.Tensor
    subject_id: torch.Tensor
    window_id: torch.Tensor
    source_epoch_index: torch.Tensor
    occurrence_index: torch.Tensor
    epoch_position: torch.Tensor
    context_summary: torch.Tensor | None = None


def evaluate_window_logits(logits, labels, log_prior, num_classes=5):
    if logits.ndim != 4 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [W,E,P,K] and labels [W,E]")
    windows, epochs, patches, classes = logits.shape
    flat_logits = logits.reshape(windows * epochs, patches, classes)
    flat_labels = labels.reshape(windows * epochs)
    curves = evaluate_per_k(
        flat_logits,
        flat_labels,
        log_prior,
        num_classes=num_classes,
    )
    ml, mapp = ml_map_predictions(flat_logits, log_prior)
    y_true = flat_labels.cpu().numpy()
    all_labels = list(range(num_classes))
    for branch, predictions in (("ml", ml), ("map", mapp)):
        predictions = predictions.cpu().numpy()
        curves[branch]["precision_per_k"] = np.stack([
            precision_score(
                y_true,
                predictions[:, patch],
                labels=all_labels,
                average=None,
                zero_division=0,
            )
            for patch in range(patches)
        ])
    summary = summary_at_k(
        flat_logits,
        flat_labels,
        log_prior,
        num_classes=num_classes,
        k_index=-1,
    )
    return curves, summary


@torch.no_grad()
def collect_outputs(
    model,
    loader,
    device,
    num_patches,
    history_table=None,
    return_summary=False,
):
    model.eval()
    parts = {
        name: []
        for name in (
            "logits",
            "labels",
            "end_idx",
            "subject_id",
            "window_id",
            "source_epoch_index",
            "occurrence_index",
            "epoch_position",
        )
    }
    summary_parts = []
    for batch in loader:
        x = batch["x"].float().to(device, non_blocking=True)
        window_id = batch["window_id"].long()
        if history_table is None:
            end_idx = torch.full(
                (x.shape[0], x.shape[1]),
                num_patches - 1,
                dtype=torch.long,
                device=device,
            )
        else:
            end_idx = history_table[window_id].to(device=device)
        model_out = model(x, end_idx, return_summary=return_summary)
        if return_summary:
            logits, summary = model_out
            summary_parts.append(summary.cpu())
        else:
            logits = model_out
        parts["logits"].append(logits.cpu())
        parts["labels"].append(batch["y"].long())
        parts["end_idx"].append(end_idx.cpu())
        for name in (
            "subject_id",
            "window_id",
            "source_epoch_index",
            "occurrence_index",
            "epoch_position",
        ):
            parts[name].append(batch[name].long())
    values = {name: torch.cat(value, dim=0) for name, value in parts.items()}
    values["context_summary"] = (
        torch.cat(summary_parts, dim=0) if return_summary else None
    )
    return CollectedOutputs(**values)


def evaluate_collected(collected, log_prior, num_classes=5):
    return evaluate_window_logits(
        collected.logits,
        collected.labels,
        log_prior,
        num_classes=num_classes,
    )


def evaluate_selection(collected, num_classes=5):
    """Minimal set needed for per-epoch best selection: ML per-k macro-F1 mean
    over all positions + MF1/acc at the final position k.

    Only the ML branch is computed (uniform prior = argmax logits), using the same
    sklearn calls and same reshape order as ``evaluate_per_k``'s ``curves["ml"]``,
    so ``mean_mf1`` / final-position MF1 / final-position acc are **pointwise
    identical** to the full evaluation. MAP, per-class precision/recall/f1, and
    summary are NOT computed — these per-epoch "compute-and-discard" intermediate
    layers are recomputed once at fold-end on the best weights by
    ``evaluate_collected``. This function does not contribute to the final return
    value; it only drives early-stopping selection and the ML diagnostic numbers
    on the progress bar.
    """
    logits = collected.logits
    labels = collected.labels
    if logits.ndim != 4 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [W,E,P,K] and labels [W,E]")
    windows, epochs, patches, classes = logits.shape
    flat_logits = logits.reshape(windows * epochs, patches, classes)
    y_true = labels.reshape(windows * epochs).cpu().numpy()
    ml = flat_logits.argmax(-1).cpu().numpy()
    all_labels = list(range(num_classes))
    mf1_k = np.asarray([
        f1_score(y_true, ml[:, patch], labels=all_labels,
                 average="macro", zero_division=0)
        for patch in range(patches)
    ])
    return {
        "mean_mf1": float(mf1_k.mean()),
        "mf1_at_last": float(mf1_k[-1]),
        "acc_at_last": float(accuracy_score(y_true, ml[:, -1])),
    }


def evaluate_summary(collected, log_prior, num_classes=5):
    """Detailed per-epoch log metrics at the final position (k=T), ML/MAP each.

    Uses the same implementation and reshape as ``summary_at_k`` inside
    ``evaluate_window_logits``; only called when logger is non-null — the HPO path
    (logger=None) skips this entirely to save CPU.
    """
    logits = collected.logits
    labels = collected.labels
    if logits.ndim != 4 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [W,E,P,K] and labels [W,E]")
    windows, epochs, patches, classes = logits.shape
    return summary_at_k(
        logits.reshape(windows * epochs, patches, classes),
        labels.reshape(windows * epochs),
        log_prior,
        num_classes=num_classes,
        k_index=-1,
    )


def aggregate_random_curves(results, summaries=None):
    if not results:
        raise ValueError("results must not be empty")
    aggregate = {}
    array_keys = (
        "acc_k",
        "mf1_k",
        "precision_per_k",
        "f1_per_k",
        "recall_per_k",
    )
    for branch in ("ml", "map"):
        aggregate[branch] = {}
        for key in array_keys:
            stacked = np.stack([
                result[branch][key]
                for _, result in sorted(results.items())
            ])
            aggregate[branch][f"{key}_mean"] = stacked.mean(axis=0)
            aggregate[branch][f"{key}_std"] = stacked.std(axis=0)
        means = np.asarray([
            result[branch]["mean_mf1"]
            for _, result in sorted(results.items())
        ])
        aggregate[branch]["mean_mf1_mean"] = float(means.mean())
        aggregate[branch]["mean_mf1_std"] = float(means.std())
        if summaries is not None:
            if tuple(sorted(summaries)) != tuple(sorted(results)):
                raise ValueError("curve and summary seeds must match")
            first_seed = sorted(summaries)[0]
            for key in summaries[first_seed][branch]:
                stacked = np.stack([
                    np.asarray(summaries[seed][branch][key])
                    for seed in sorted(summaries)
                ])
                aggregate[branch][f"summary_{key}_mean"] = stacked.mean(axis=0)
                aggregate[branch][f"summary_{key}_std"] = stacked.std(axis=0)
    return aggregate
