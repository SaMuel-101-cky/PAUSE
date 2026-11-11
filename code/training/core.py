"""PAUSE training/evaluation shared kernel (dataset-agnostic).
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def lsel_loss(logits, y, class_weight):
    """LSEL = class-balanced per-prefix CE.

    logits[B,T,K], y[B] (T positions within one epoch share the same label),
    class_weight[K] ∝ 1/M_k. Implemented as a weighted mean of class-balanced CE
    (scale-invariant, same order of magnitude as standard CE).

    Invariant: every batch must contain at least one sample from a class with
    class_weight > 0 (otherwise the weighted-mean denominator is 0 → NaN).
    The current call chain satisfies this naturally: class_weight and the training
    loader are both derived from the same M_k = compute_class_counts(y_train);
    classes with zero weight never appear in a batch. If you call this function
    outside that chain, you must ensure this invariant yourself.
    """
    B, T, K = logits.shape
    targets = y.unsqueeze(1).expand(B, T).reshape(-1)      # [B*T], aligned with reshape order
    return F.cross_entropy(logits.reshape(B * T, K), targets,
                           weight=class_weight, reduction="mean")


from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, cohen_kappa_score, confusion_matrix)


def ml_map_predictions(logits, log_prior):
    """logits[...,K], log_prior[K] → (ml, map) decision indices, shape logits[...].

    ML = argmax s (uniform prior); MAP = argmax(s + log p̂) (log-domain formulation).
    """
    ml = logits.argmax(-1)
    mapp = (logits + log_prior).argmax(-1)
    return ml, mapp


def evaluate_per_k(logits, labels, log_prior, num_classes=5):
    """Per-prefix-length k=1..T ML/MAP metric curves.

    logits[N,T,K], labels[N], log_prior[K]. Position p corresponds to prefix length k=p+1.
    sklearn is called uniformly with labels=range(K) to avoid misaligned per-class arrays
    when a test set is missing some classes.
    """
    N, T, K = logits.shape
    assert K == num_classes, f"logits last dim {K} != num_classes {num_classes}"
    ml, mapp = ml_map_predictions(logits, log_prior)
    y_true = labels.cpu().numpy()
    all_labels = list(range(num_classes))
    out = {}
    for name, preds in (("ml", ml), ("map", mapp)):
        acc_k, mf1_k, f1_k, rec_k = [], [], [], []
        preds_np = preds.cpu().numpy()
        for p in range(T):
            yp = preds_np[:, p]
            acc_k.append(accuracy_score(y_true, yp))
            mf1_k.append(f1_score(y_true, yp, labels=all_labels, average="macro", zero_division=0))
            f1_k.append(f1_score(y_true, yp, labels=all_labels, average=None, zero_division=0))
            rec_k.append(recall_score(y_true, yp, labels=all_labels, average=None, zero_division=0))
        mf1_k_arr = np.asarray(mf1_k)
        out[name] = {
            "acc_k": np.asarray(acc_k),
            "mf1_k": mf1_k_arr,
            "f1_per_k": np.asarray(f1_k),
            "recall_per_k": np.asarray(rec_k),
            "mean_mf1": float(mf1_k_arr.mean()),   # macro-F1 averaged over k=1..T (best-selection criterion)
        }
    return out


def _gmean_and_cm(y_true, y_pred, recall_per_class, num_classes):
    """Per-class G-mean = sqrt(specificity·recall) + confusion matrix.

    Called uniformly with labels=range(K); rows/columns remain complete when classes
    are missing, avoiding misalignment with recall_per_class.
    specificity is set to 0 when the denominator is 0 (zero_division semantics).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    denom = tn + fp
    specificity = np.divide(tn, denom, out=np.zeros_like(tn), where=denom > 0)
    gmean_pc = np.sqrt(specificity * np.asarray(recall_per_class, dtype=np.float64))
    return gmean_pc, float(np.mean(gmean_pc)), cm


def summary_at_k(logits, labels, log_prior, num_classes=5, k_index=-1):
    """Detailed metrics at a single prefix length (default final position k=T),
    consistent with the reference implementation, one set each for ML and MAP.

    Returns {branch: {acc, mf1, kappa, mgm, precision_per_class, recall_per_class,
                      f1_per_class, gmean_per_class, confusion_matrix}}.
    acc = overall accuracy (multiclass = micro-precision of the reference implementation);
    per_class arrays always have length num_classes.
    """
    ml, mapp = ml_map_predictions(logits, log_prior)
    y_true = labels.cpu().numpy()
    all_labels = list(range(num_classes))
    out = {}
    for name, preds in (("ml", ml), ("map", mapp)):
        yp = preds.cpu().numpy()[..., k_index]
        recall_pc = recall_score(y_true, yp, labels=all_labels, average=None, zero_division=0)
        gmean_pc, mgm, cm = _gmean_and_cm(y_true, yp, recall_pc, num_classes)
        out[name] = {
            "acc": float(accuracy_score(y_true, yp)),
            "mf1": float(f1_score(y_true, yp, labels=all_labels, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y_true, yp, labels=all_labels)),
            "mgm": mgm,
            "precision_per_class": precision_score(y_true, yp, labels=all_labels, average=None, zero_division=0),
            "recall_per_class": recall_pc,
            "f1_per_class": f1_score(y_true, yp, labels=all_labels, average=None, zero_division=0),
            "gmean_per_class": gmean_pc,
            "confusion_matrix": cm,
        }
    return out


def log_summary_block(logger, tag, s, num_classes=5):
    """Print overall + per-class metrics for one branch (ML or MAP) at k=30,
    consistent with the reference log format.

    Prec = per-class precision, acc = overall accuracy.
    Shared by train_one_fold (per-epoch) and main (fold-end / aggregation).
    """
    logger.info(f"  [{tag}] acc={s['acc']:.4f} MF1={s['mf1']:.4f} "
                f"Kappa={s['kappa']:.4f} MGm={s['mgm']:.4f}")
    for i in range(num_classes):
        logger.info(f"    [{tag}] Class {i + 1}: Prec={s['precision_per_class'][i]:.4f} "
                    f"Recall={s['recall_per_class'][i]:.4f} "
                    f"F1={s['f1_per_class'][i]:.4f} G-mean={s['gmean_per_class'][i]:.4f}")


import csv
import os
import copy
from torch.optim import lr_scheduler
from tqdm import tqdm
from lion_pytorch import Lion
from dataset.core import class_weights_from_counts, priors_from_counts, make_loaders


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CosineAnnealingWarmupRestarts(lr_scheduler._LRScheduler):
    """Cosine schedule with warmup (following the reference implementation)."""
    def __init__(self, optimizer, first_cycle_steps, warmup_steps=0,
                 max_lr=1e-3, min_lr=1e-6, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch / self.warmup_steps
                    for _ in self.optimizer.param_groups]
        progress = (self.last_epoch - self.warmup_steps) / (self.first_cycle_steps - self.warmup_steps)
        return [self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for _ in self.optimizer.param_groups]


@torch.no_grad()
def evaluate_fold(model, test_iter, log_prior, device, num_classes=5,
                  return_logits=False):
    model.eval()
    logits_all, y_all = [], []
    for xb, yb in test_iter:
        xb = xb.float().to(device)
        logits_all.append(model(xb).cpu())
        y_all.append(yb)
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(y_all, 0)
    lp = log_prior.cpu()
    curves = evaluate_per_k(logits, labels, lp, num_classes)
    summary = summary_at_k(logits, labels, lp, num_classes, k_index=-1)  # k=30 detailed metrics
    if return_logits:
        return curves, summary, (logits, labels)
    return curves, summary


def save_curves(curves, out_prefix):
    """Write .npz (full curves) + .csv (k, ml/map acc/mf1)."""
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.savez(
        out_prefix + ".npz",
        ml_acc=curves["ml"]["acc_k"], ml_mf1=curves["ml"]["mf1_k"],
        ml_f1=curves["ml"]["f1_per_k"], ml_recall=curves["ml"]["recall_per_k"],
        map_acc=curves["map"]["acc_k"], map_mf1=curves["map"]["mf1_k"],
        map_f1=curves["map"]["f1_per_k"], map_recall=curves["map"]["recall_per_k"],
        ml_mean_mf1=curves["ml"]["mean_mf1"], map_mean_mf1=curves["map"]["mean_mf1"],
    )
    T = len(curves["ml"]["acc_k"])
    with open(out_prefix + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "ml_acc", "ml_mf1", "map_acc", "map_mf1"])
        for p in range(T):
            w.writerow([p + 1,
                        curves["ml"]["acc_k"][p], curves["ml"]["mf1_k"][p],
                        curves["map"]["acc_k"][p], curves["map"]["mf1_k"][p]])

