import torch
from config.gp import GPConfig

def calc_sns(labels, predictions, num_classes_):
    """Compute the macro average of per-class sensitivity (recall).

    Args:
        labels: Tensor of shape (N,) with integer class labels.
        predictions: Tensor of shape (N,) with integer predicted classes.

    Returns:
        scalar tensor: macro-averaged recall (sensitivity).
    """
    from sklearn.metrics import recall_score
    num_classes = num_classes_
    return torch.tensor(
        recall_score(
            labels.cpu().numpy(), predictions.cpu().numpy(),
            labels=list(range(num_classes)), average="macro", zero_division=0,
        ),
        dtype=torch.float32,
    )