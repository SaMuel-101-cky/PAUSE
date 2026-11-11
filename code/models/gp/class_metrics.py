"""Per-class metrics shared by GP evaluation outputs."""

import numpy as np


def summarize_per_class_outputs(
    labels, predictions, hitting_time, num_classes
) -> dict[str, dict[str, int | float | None]]:
    """Summarize prediction counts, rates, and stopping times by true class."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    hitting_time = np.asarray(hitting_time, dtype=np.int64).reshape(-1)

    if not (len(labels) == len(predictions) == len(hitting_time)):
        raise ValueError("labels, predictions, and hitting_time must have equal lengths")

    summary = {}
    for class_id in range(num_classes):
        true_class = labels == class_id
        predicted_class = predictions == class_id
        true_positive = true_class & predicted_class
        support = int(true_class.sum())
        predicted_count = int(predicted_class.sum())
        true_positive_count = int(true_positive.sum())
        recall = true_positive_count / support if support else 0.0

        summary[str(class_id)] = {
            "support": support,
            "predicted_count": predicted_count,
            "true_positive_count": true_positive_count,
            "mean_hitting_time": (
                float(hitting_time[true_class].mean()) if support else None
            ),
            "recall": recall,
            "precision": (
                true_positive_count / predicted_count if predicted_count else 0.0
            ),
            "mean_error_rate": 1.0 - recall,
        }

    return summary
