"""PAUSE cross-epoch data windowing and DataLoader."""
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.core import (
    compute_class_counts,
    load_single_epoch_file,
    load_single_epoch_labels,
    zscore_apply,
    zscore_fit,
)


@dataclass(frozen=True)
class WindowBundle:
    x: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    window_id: np.ndarray
    source_epoch_index: np.ndarray
    occurrence_index: np.ndarray
    epoch_position: np.ndarray

    def __post_init__(self):
        if self.x.ndim != 4:
            raise ValueError(f"x must be [W,E,P,F], got {self.x.shape}")
        windows, epochs = self.x.shape[:2]
        expected = (windows, epochs)
        for name in (
            "y",
            "subject_id",
            "source_epoch_index",
            "occurrence_index",
            "epoch_position",
        ):
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} must be {expected}, got {value.shape}")
        if self.window_id.shape != (windows,):
            raise ValueError(f"window_id must be [{windows}]")


def _validate_subject_arrays(x, y, num_epochs):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if x.ndim != 3:
        raise ValueError(f"x must be [N,P,F], got {x.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x/y epoch count mismatch")
    if num_epochs < 1 or x.shape[0] < num_epochs:
        raise ValueError("subject must contain at least one complete window")
    return x, y


def _build_bundle(x, y, source_idx, subject_id, num_epochs):
    windows = source_idx.shape[0]
    flat_idx = source_idx.reshape(-1)
    occurrence = np.zeros_like(flat_idx)
    counts = {}
    for i, source in enumerate(flat_idx.tolist()):
        occurrence[i] = counts.get(source, 0)
        counts[source] = occurrence[i] + 1
    occurrence = occurrence.reshape(windows, num_epochs)
    positions = np.broadcast_to(
        np.arange(num_epochs, dtype=np.int64),
        (windows, num_epochs),
    ).copy()
    return WindowBundle(
        x=x[flat_idx].reshape(windows, num_epochs, *x.shape[1:]),
        y=y[flat_idx].reshape(windows, num_epochs),
        subject_id=np.full((windows, num_epochs), subject_id, dtype=np.int64),
        window_id=np.arange(windows, dtype=np.int64),
        source_epoch_index=source_idx.astype(np.int64, copy=False),
        occurrence_index=occurrence,
        epoch_position=positions,
    )


def make_train_windows(x, y, subject_id, num_epochs=32):
    x, y = _validate_subject_arrays(x, y, num_epochs)
    usable = (x.shape[0] // num_epochs) * num_epochs
    source_idx = np.arange(usable, dtype=np.int64).reshape(-1, num_epochs)
    return _build_bundle(x, y, source_idx, subject_id, num_epochs)


def make_legacy_test_windows(x, y, subject_id, num_epochs=32):
    x, y = _validate_subject_arrays(x, y, num_epochs)
    count = x.shape[0]
    full_end = (count // num_epochs) * num_epochs
    remainder = count % num_epochs
    if remainder == 0:
        flat_idx = np.arange(count, dtype=np.int64)
    else:
        borrow = num_epochs - remainder
        tail_idx = np.concatenate([
            np.arange(full_end - borrow, full_end, dtype=np.int64),
            np.arange(full_end, count, dtype=np.int64),
        ])
        flat_idx = np.concatenate([
            np.arange(full_end, dtype=np.int64),
            tail_idx,
        ])
    source_idx = flat_idx.reshape(-1, num_epochs)
    return _build_bundle(x, y, source_idx, subject_id, num_epochs)


@dataclass(frozen=True)
class PauseFold:
    train: WindowBundle
    test: WindowBundle
    class_counts: np.ndarray
    train_mean: float
    train_std: float


@dataclass(frozen=True)
class PauseCalibrationFold:
    """Calibration retains only test windows and reuses training statistics from the checkpoint."""
    test: WindowBundle
    class_counts: np.ndarray
    train_mean: float
    train_std: float


def concatenate_bundles(bundles):
    bundles = tuple(bundles)
    if not bundles:
        raise ValueError("bundles must not be empty")
    fields = (
        "x",
        "y",
        "subject_id",
        "window_id",
        "source_epoch_index",
        "occurrence_index",
        "epoch_position",
    )
    values = {
        field: np.concatenate([getattr(bundle, field) for bundle in bundles], axis=0)
        for field in fields
    }
    return WindowBundle(**values)


def _normalized(bundle, mean, std):
    if not np.isfinite(std) or std <= 0:
        raise ValueError("training std must be finite and positive")
    return replace(
        bundle,
        x=zscore_apply(bundle.x, mean, std).astype(np.float32, copy=False),
    )


def _renumber_window_ids(bundle):
    """After concatenating test bundles from multiple files, renumber window_id
    to global arange(W).

    make_legacy_test_windows already assigns arange for single-file case;
    multi-file concatenation produces duplicates, and the calibration random
    history table indexes by num_windows, so global uniqueness is required.
    """
    return replace(
        bundle,
        window_id=np.arange(bundle.x.shape[0], dtype=np.int64),
    )


def assemble_fold(fold, num_epochs=32, num_classes=5):
    """Assemble a PauseFold from a Fold (train/test file partition).

    Train files go through make_train_windows; test files go through
    make_legacy_test_windows and are concatenated (window_id renumbered for
    multi-file). zscore is fitted only on training windows.
    """
    train_bundles = []
    for path in fold.train_paths:
        x, y = load_single_epoch_file(str(path))
        train_bundles.append(
            make_train_windows(x, y, fold.subject_index[path], num_epochs=num_epochs)
        )
    test_bundles = []
    for path in fold.test_paths:
        x, y = load_single_epoch_file(str(path))
        test_bundles.append(
            make_legacy_test_windows(x, y, fold.subject_index[path], num_epochs=num_epochs)
        )
    train_bundle = concatenate_bundles(train_bundles)
    test_bundle = _renumber_window_ids(concatenate_bundles(test_bundles))
    mean, std = zscore_fit(train_bundle.x)
    train_bundle = _normalized(train_bundle, mean, std)
    test_bundle = _normalized(test_bundle, mean, std)
    counts = compute_class_counts(train_bundle.y.reshape(-1), num_classes=num_classes)
    return PauseFold(
        train=train_bundle,
        test=test_bundle,
        class_counts=counts,
        train_mean=mean,
        train_std=std,
    )


def assemble_calibration_fold(
    fold,
    *,
    train_mean,
    train_std,
    num_epochs=32,
    num_classes=5,
):
    """Assemble calibration test fold using training checkpoint statistics.

    Training side only reads labels and computes ``class_counts`` following the
    tail truncation rule of ``make_train_windows``; full PSD is only read for
    test files.
    """
    train_mean = float(train_mean)
    train_std = float(train_std)
    if not np.isfinite(train_mean) or not np.isfinite(train_std) or train_std <= 0:
        raise ValueError("checkpoint normalization stats must be finite and positive")

    class_counts = np.zeros(num_classes, dtype=np.int64)
    for path in fold.train_paths:
        y = load_single_epoch_labels(str(path))
        usable = (y.shape[0] // num_epochs) * num_epochs
        if usable < num_epochs:
            raise ValueError("subject must contain at least one complete window")
        class_counts += compute_class_counts(y[:usable], num_classes=num_classes)

    test_bundles = []
    for path in fold.test_paths:
        x, y = load_single_epoch_file(str(path))
        test_bundles.append(
            make_legacy_test_windows(
                x, y, fold.subject_index[path], num_epochs=num_epochs
            )
        )
    test_bundle = _renumber_window_ids(concatenate_bundles(test_bundles))
    test_bundle = _normalized(test_bundle, train_mean, train_std)
    return PauseCalibrationFold(
        test=test_bundle,
        class_counts=class_counts,
        train_mean=train_mean,
        train_std=train_std,
    )


def assemble_dreams_fold(subject_paths, test_subject, num_epochs=32):
    """DREAMS LOSO thin wrapper (kept for hpo/loso.py and existing tests)."""
    if test_subject not in subject_paths:
        raise ValueError(f"missing test subject {test_subject}")
    from dataset.folds import Fold
    subject_index = {path: sid for sid, path in subject_paths.items()}
    ordered = sorted(subject_paths)
    fold = Fold(
        fold_id=test_subject,
        train_paths=[subject_paths[s] for s in ordered if s != test_subject],
        test_paths=[subject_paths[test_subject]],
        test_units=[test_subject],
        subject_index=subject_index,
    )
    return assemble_fold(fold, num_epochs=num_epochs)


class PauseWindowDataset(Dataset):
    def __init__(self, bundle):
        self.bundle = bundle

    def __len__(self):
        return self.bundle.x.shape[0]

    def __getitem__(self, index):
        return {
            "x": torch.as_tensor(self.bundle.x[index], dtype=torch.float32),
            "y": torch.as_tensor(self.bundle.y[index], dtype=torch.long),
            "subject_id": torch.as_tensor(
                self.bundle.subject_id[index], dtype=torch.long
            ),
            "window_id": torch.as_tensor(
                self.bundle.window_id[index], dtype=torch.long
            ),
            "source_epoch_index": torch.as_tensor(
                self.bundle.source_epoch_index[index], dtype=torch.long
            ),
            "occurrence_index": torch.as_tensor(
                self.bundle.occurrence_index[index], dtype=torch.long
            ),
            "epoch_position": torch.as_tensor(
                self.bundle.epoch_position[index], dtype=torch.long
            ),
        }


def _seed_worker(worker_id, seed):
    np.random.seed(seed + worker_id)


def make_window_loaders(
    train_bundle,
    test_bundle,
    batch_size_windows,
    seed=2025,
    num_workers=4,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_init = partial(_seed_worker, seed=seed)
    train_loader = DataLoader(
        PauseWindowDataset(train_bundle),
        batch_size=batch_size_windows,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=worker_init,
    )
    test_loader = DataLoader(
        PauseWindowDataset(test_bundle),
        batch_size=batch_size_windows,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init,
    )
    return train_loader, test_loader
