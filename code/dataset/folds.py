"""Cross-epoch PAUSE fold enumeration — single source of truth for training and calibration.

Three split schemes mirror the reference implementation, including random.seed(2025)
and individual remainder rules. EDF and SHHS remainder rules differ, so they are
implemented separately and cannot be merged.
"""
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_paths: list      # list[Path]
    test_paths: list       # list[Path] (multiple files in k-fold)
    test_units: list       # list[int], test unit integer id (subject id / file id), provenance
    subject_index: dict    # {Path: int}, per-file → stable integer subject id (fed to WindowBundle.subject_id)


def _iter_mat(data_dir, prefix):
    """Return sorted paths, filtering macOS ._ metadata; prefix matches the reference implementation entry points."""
    data_dir = Path(data_dir)
    return [
        p for p in sorted(data_dir.iterdir())
        if p.name.startswith(prefix) and p.name.endswith(".mat")
    ]


def _loso_folds(cfg):
    # DREAMS: subject{n}.mat, 1 file = 1 subject, LOSO; fold_id = subject index.
    paths = {}
    for p in _iter_mat(cfg.data_dir, "subject"):
        sid = int(p.stem.removeprefix("subject"))
        paths[sid] = p
    subject_index = {p: sid for sid, p in paths.items()}
    ordered = sorted(paths)
    folds = []
    for sid in ordered:
        test_paths = [paths[sid]]
        train_paths = [paths[o] for o in ordered if o != sid]
        folds.append(Fold(sid, train_paths, test_paths, [sid], subject_index))
    return folds


def _grouped_kfold_folds(cfg):
    # EDF20/78: group subjects by filename chars [3:5]; shuffle → equal-sized slices → append remainders to the end.
    subject_files = defaultdict(list)
    for p in _iter_mat(cfg.data_dir, "SC4"):
        subject_files[p.name[3:5]].append(p)
    subject_index = {
        p: int(key) for key, plist in subject_files.items() for p in plist
    }
    subjects = sorted(subject_files.keys())
    random.seed(2025)
    random.shuffle(subjects)
    fold_size = len(subjects) // cfg.num_folds
    grouped = [subjects[i * fold_size:(i + 1) * fold_size] for i in range(cfg.num_folds)]
    for i, s in enumerate(subjects[cfg.num_folds * fold_size:]):
        grouped[i].append(s)
    folds = []
    for fid, subs in enumerate(grouped):
        subs_set = set(subs)
        test_paths = [f for s in subs for f in subject_files[s]]
        train_paths = [f for s in subjects if s not in subs_set for f in subject_files[s]]
        test_units = sorted(int(s) for s in subs)
        folds.append(Fold(fid, train_paths, test_paths, test_units, subject_index))
    return folds


def _file_kfold_folds(cfg):
    # SHHS: shhs*.mat, 1 file = 1 subject; shuffle → first r folds get contiguous variable-length slices (size+1).
    files = _iter_mat(cfg.data_dir, "shhs")
    subject_index = {p: int(p.stem.split("-")[1]) for p in files}
    random.seed(2025)
    shuffled = files.copy()
    random.shuffle(shuffled)
    fold_size = len(shuffled) // cfg.num_folds
    remainder = len(shuffled) % cfg.num_folds
    folds, start = [], 0
    for fid in range(cfg.num_folds):
        size = fold_size + (1 if fid < remainder else 0)
        test_paths = shuffled[start:start + size]
        start += size
        test_set = set(test_paths)
        train_paths = [f for f in files if f not in test_set]
        test_units = sorted(subject_index[p] for p in test_paths)
        folds.append(Fold(fid, train_paths, test_paths, test_units, subject_index))
    return folds


_SCHEMES = {
    "loso": _loso_folds,
    "grouped_kfold": _grouped_kfold_folds,
    "file_kfold": _file_kfold_folds,
}


def enumerate_folds(cfg):
    if cfg.fold_scheme not in _SCHEMES:
        raise ValueError(f"unknown fold_scheme '{cfg.fold_scheme}'")
    folds = _SCHEMES[cfg.fold_scheme](cfg)
    if not folds or any(not f.test_paths for f in folds):
        raise ValueError(f"{cfg.name}: empty fold(s); check data_dir {cfg.data_dir}")
    return folds
