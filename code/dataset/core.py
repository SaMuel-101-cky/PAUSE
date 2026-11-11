"""PAUSE single-epoch data pipeline and class statistics.
"""
from functools import partial

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


def _seed_worker(wid, seed):
    """DataLoader worker_init_fn: seed numpy RNG per worker with seed+wid.

    Uses a named function (not a lambda closure) so it can be pickled under
    the spawn multiprocessing start method.
    """
    np.random.seed(seed + wid)


def load_single_epoch_file(path, data_key="PSD100", label_key="label"):
    """Load all epochs from one .mat file as independent samples.

    Returns X[N,30,128] float32, y[N] int64. No windowing / sliding / transition.
    """
    d = loadmat(path)
    X = np.array(d[data_key], dtype=np.float32)          # [N, 30, 128]
    y = np.array(d[label_key], dtype=np.int64).flatten()  # [N]
    assert X.shape[0] == y.shape[0], f"{path}: X/y count mismatch"
    return X, y


def load_single_epoch_labels(path, label_key="label"):
    """Read only the labels from one ``.mat`` file, avoiding PSD decoding on the training side during calibration."""
    d = loadmat(path, variable_names=[label_key])
    return np.array(d[label_key], dtype=np.int64).flatten()


def load_dataset_files(files, data_key="PSD100", label_key="label"):
    """Concatenate all epochs from multiple files into a single dataset."""
    Xs, ys = [], []
    for f in files:
        X, y = load_single_epoch_file(f, data_key, label_key)
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def zscore_fit(X):
    """Global scalar z-score (following the reference implementation convention: overall mean/std)."""
    return float(np.mean(X)), float(np.std(X))


def zscore_apply(X, mean, std):
    return (X - mean) / std


def compute_class_counts(y, num_classes=5):
    """Per-class sample counts M_k for a training fold."""
    return np.bincount(np.asarray(y).ravel(), minlength=num_classes)[:num_classes]


def class_weights_from_counts(counts):
    """LSEL class-balanced weights ∝ 1/M_k; classes with M_k=0 get weight 0 (excluded from loss).

    Only proportionality to 1/M_k is required (the global scale is absorbed
    by the weighted-mean normalization).
    """
    counts = np.asarray(counts, dtype=np.float64)
    w = np.zeros_like(counts)
    nz = counts > 0
    w[nz] = 1.0 / counts[nz]
    return torch.tensor(w, dtype=torch.float32)


def priors_from_counts(counts, eps=1e-6):
    """Empirical class prior p̂_k = M_k/M; missing classes are floored to eps then renormalized."""
    counts = np.asarray(counts, dtype=np.float64)
    p = counts / max(counts.sum(), 1.0)
    p = np.clip(p, eps, None)
    p = p / p.sum()
    return torch.tensor(p, dtype=torch.float32)


class SingleEpochDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loaders(X_train, y_train, X_test, y_test, batch_size, seed=2025, num_workers=4):
    g = torch.Generator()
    g.manual_seed(seed)
    Xtr = torch.as_tensor(X_train, dtype=torch.float32)
    ytr = torch.as_tensor(y_train, dtype=torch.long)
    Xte = torch.as_tensor(X_test, dtype=torch.float32)
    yte = torch.as_tensor(y_test, dtype=torch.long)
    train_iter = DataLoader(
        SingleEpochDataset(Xtr, ytr), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=g,
        worker_init_fn=partial(_seed_worker, seed=seed),
    )
    test_iter = DataLoader(
        SingleEpochDataset(Xte, yte), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=partial(_seed_worker, seed=seed),
    )
    return train_iter, test_iter
