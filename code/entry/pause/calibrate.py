"""PAUSE cross-epoch, variant-aware per-k temperature calibration entry point."""
import argparse
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibration.core import (
    FoldLogits,
    calibrated_log_probs,
    evaluate_fold_calibration,
    fit_target_temperatures,
    reliability_bins,
)
from config.pause import (
    VARIANTS,
    cache_stem_fmt,
    calib_dir as cfg_calib_dir,
    ckpt_dir as cfg_ckpt_dir,
    get_cfg,
)
from dataset.folds import Fold, enumerate_folds
from dataset.pause import (
    PauseWindowDataset,
    assemble_calibration_fold,
    assemble_dreams_fold,
    assemble_fold,
)

from entry.pause.history import (
    build_random_history_tables,
    history_end_idx_dist,
)
from models.pause.model import PAUSE


SUBJECT_IDS = tuple(range(1, 21))
NUM_EPOCHS = 32
NUM_CLASSES = 5
BATCH_SIZE_WINDOWS = 16
DEFAULT_DATA_DIR = Path("dataset/DREAMS")
DEFAULT_CKPT_DIR = Path("checkpoint/pause/DREAMS")
DEFAULT_OUTPUT_DIR = Path("outputs/calibration/pause/DREAMS")
DEFAULT_SEED_START = 2025
DEFAULT_NUM_HISTORY_DRAWS = 32
MODEL_VARIANTS = tuple(VARIANTS)

LOGGER = logging.getLogger("pause_temperature_calibration")


def _subject_stem(subject_id):
    return f"subject{subject_id}"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(subject_paths):
    digest = hashlib.sha256()
    for subject_id, path in sorted(subject_paths.items()):
        digest.update(str(subject_id).encode("utf-8"))
        digest.update(Path(path).name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".npz"
    )
    os.close(fd)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_config_json(model_kwargs, num_classes):
    return json.dumps(
        {"model_kwargs": model_kwargs, "num_classes": num_classes},
        sort_keys=True,
        separators=(",", ":"),
    )


def load_fold_cache(
    path,
    *,
    expected_subject_id=None,
    expected_checkpoint_hash=None,
    expected_data_fingerprint=None,
):
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "logits", "labels", "log_prior", "class_counts", "train_mean",
            "train_std", "subject_id", "checkpoint_hash",
            "data_fingerprint", "model_config_json",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"cache missing keys: {sorted(missing)}")
        subject_id = int(saved["subject_id"].item())
        checkpoint_hash = str(saved["checkpoint_hash"].item())
        data_fingerprint = str(saved["data_fingerprint"].item())
        if expected_subject_id is not None and subject_id != expected_subject_id:
            raise ValueError("cache subject id mismatch")
        if (
            expected_checkpoint_hash is not None
            and checkpoint_hash != expected_checkpoint_hash
        ):
            raise ValueError("cache checkpoint hash mismatch")
        if (
            expected_data_fingerprint is not None
            and data_fingerprint != expected_data_fingerprint
        ):
            raise ValueError("cache data fingerprint mismatch")
        fold = FoldLogits(
            subject_id=subject_id,
            logits=torch.from_numpy(saved["logits"].copy()).double(),
            labels=torch.from_numpy(saved["labels"].copy()).long(),
            log_prior=torch.from_numpy(saved["log_prior"].copy()).double(),
        )
        metadata = {
            "class_counts": saved["class_counts"].copy(),
            "train_mean": float(saved["train_mean"].item()),
            "train_std": float(saved["train_std"].item()),
            "checkpoint_hash": checkpoint_hash,
            "data_fingerprint": data_fingerprint,
            "model_config_json": str(saved["model_config_json"].item()),
        }
    return fold, metadata


def _load_all_fold_caches(cache_dir, subject_ids, stem_for_id=_subject_stem):
    folds = []
    for subject_id in subject_ids:
        path = Path(cache_dir) / f"{stem_for_id(subject_id)}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        fold, _ = load_fold_cache(path, expected_subject_id=subject_id)
        folds.append(fold)
    return folds


def run_fit_stage(
    *,
    cache_dir,
    temperature_dir,
    metrics_dir,
    subject_ids=SUBJECT_IDS,
    n_bins=15,
    stem_for_id=_subject_stem,
):
    folds = _load_all_fold_caches(cache_dir, subject_ids, stem_for_id)
    fold_by_id = {fold.subject_id: fold for fold in folds}
    if sorted(fold_by_id) != sorted(subject_ids):
        raise ValueError("cache subject set does not match requested subject set")
    for target_subject in subject_ids:
        fit = fit_target_temperatures(folds, target_subject)
        expected_sources = sorted(set(subject_ids) - {target_subject})
        if fit.source_subjects.tolist() != expected_sources:
            raise AssertionError("target subject leaked into temperature sources")
        atomic_savez(
            Path(temperature_dir) / f"{stem_for_id(target_subject)}.npz",
            target_subject=np.asarray(target_subject, dtype=np.int64),
            source_subjects=fit.source_subjects,
            temperature=fit.temperature,
            nll_fit_before=fit.nll_fit_before,
            nll_fit_after=fit.nll_fit_after,
            optimizer_status=fit.optimizer_status,
            fallback_used=fit.fallback_used,
        )
        metrics = evaluate_fold_calibration(
            fold_by_id[target_subject],
            fit.temperature,
            n_bins=n_bins,
            n1_index=1,
        )
        atomic_savez(
            Path(metrics_dir) / f"{stem_for_id(target_subject)}.npz",
            target_subject=np.asarray(target_subject, dtype=np.int64),
            **metrics,
        )


def validate_complete_run(
    *,
    temperature_dir,
    metrics_dir,
    subject_ids=SUBJECT_IDS,
    tolerance=1e-10,
    stem_for_id=_subject_stem,
):
    for target in subject_ids:
        stem = stem_for_id(target)
        temperature_path = Path(temperature_dir) / f"{stem}.npz"
        metrics_path = Path(metrics_dir) / f"{stem}.npz"
        if not temperature_path.is_file() or not metrics_path.is_file():
            raise ValueError(f"missing output for target subject {target}")
        with np.load(temperature_path, allow_pickle=False) as saved:
            expected = sorted(set(subject_ids) - {target})
            actual = saved["source_subjects"].tolist()
            if actual != expected:
                raise ValueError(
                    f"subject{target} source_subjects {actual} != {expected}"
                )
            temperature = saved["temperature"]
            before = saved["nll_fit_before"]
            after = saved["nll_fit_after"]
            if (
                not np.isfinite(temperature).all()
                or not np.all(temperature > 0)
            ):
                raise ValueError(f"subject{target} invalid temperature")
            if np.any(after > before + tolerance):
                raise ValueError(f"subject{target} fitting NLL increased")
            if np.any(
                np.isclose(temperature, 0.05)
                | np.isclose(temperature, 20.0)
            ):
                LOGGER.warning("subject%s temperature boundary reached", target)
        with np.load(metrics_path, allow_pickle=False) as saved:
            if not bool(saved["ml_predictions_equal"].item()):
                raise ValueError(f"subject{target} ML predictions changed")


def _read_npz_series(directory, subject_ids, stem_for_id=_subject_stem):
    records = []
    for subject_id in subject_ids:
        path = Path(directory) / f"{stem_for_id(subject_id)}.npz"
        with np.load(path, allow_pickle=False) as saved:
            records.append({key: saved[key].copy() for key in saved.files})
    return records


def write_summary_csv(
    metrics_dir, output_path, subject_ids=SUBJECT_IDS, stem_for_id=_subject_stem
):
    records = _read_npz_series(metrics_dir, subject_ids, stem_for_id)
    excluded = {"target_subject", "ml_predictions_equal"}
    metric_names = sorted(
        key
        for key in records[0]
        if key not in excluded and records[0][key].ndim == 1
    )
    lines = ["metric,k,mean,std\n"]
    for metric in metric_names:
        matrix = np.stack([record[metric] for record in records])
        for k, (mean, std) in enumerate(
            zip(matrix.mean(0), matrix.std(0)), start=1
        ):
            lines.append(f"{metric},{k},{mean},{std}\n")
    atomic_write_text(output_path, "".join(lines))


def write_temperature_curve_csv(
    temperature_dir, output_path, subject_ids=SUBJECT_IDS, stem_for_id=_subject_stem
):
    records = _read_npz_series(temperature_dir, subject_ids, stem_for_id)
    matrix = np.stack([record["temperature"] for record in records])
    lines = ["metric,k,mean,std\n"]
    for k, (mean, std) in enumerate(
        zip(matrix.mean(0), matrix.std(0)), start=1
    ):
        lines.append(f"temperature,{k},{mean},{std}\n")
    atomic_write_text(output_path, "".join(lines))

def _atomic_save_figure(fig, path, *, dpi=180):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}-", suffix=path.suffix
    )
    os.close(fd)
    try:
        fig.savefig(temporary, dpi=dpi)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_figures(
    *,
    cache_dir,
    temperature_dir,
    metrics_dir,
    figure_dir,
    subject_ids=SUBJECT_IDS,
    n_bins=15,
    min_bin_count=20,
    stem_for_id=_subject_stem,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_npz_series(metrics_dir, subject_ids, stem_for_id)
    temperatures = _read_npz_series(temperature_dir, subject_ids, stem_for_id)
    x = np.arange(1, metrics[0]["ece_before"].shape[0] + 1)

    for stem, before_key, after_key, ylabel in (
        ("ece_per_k", "ece_before", "ece_after", "ECE"),
        ("nll_per_k", "nll_before", "nll_after", "NLL"),
        ("n1_ece_per_k", "n1_ece_before", "n1_ece_after", "N1 ECE"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, key in (("Before", before_key), ("After", after_key)):
            matrix = np.stack([record[key] for record in metrics])
            mean, std = matrix.mean(0), matrix.std(0)
            ax.plot(x, mean, label=label)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set(xlabel="Prefix length k", ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        _atomic_save_figure(fig, figure_dir / f"{stem}.png")
        plt.close(fig)

    matrix = np.stack([record["temperature"] for record in temperatures])
    mean, std = matrix.mean(0), matrix.std(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, mean)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Prefix length k", ylabel="Temperature")
    fig.tight_layout()
    _atomic_save_figure(fig, figure_dir / "temperature_per_k.png")
    plt.close(fig)

    folds = {
        sid: load_fold_cache(
            Path(cache_dir) / f"{stem_for_id(sid)}.npz",
            expected_subject_id=sid,
        )[0]
        for sid in subject_ids
    }
    max_steps = min(30, folds[subject_ids[0]].logits.shape[1])
    for k_one_based in (1, 5, 10, 20, max_steps):
        k = k_one_based - 1
        before_conf, before_correct = [], []
        after_conf, after_correct = [], []
        for sid in subject_ids:
            with np.load(
                Path(temperature_dir) / f"{stem_for_id(sid)}.npz",
                allow_pickle=False,
            ) as saved:
                temperature = saved["temperature"].copy()
            fold = folds[sid]
            before = calibrated_log_probs(
                fold.logits,
                fold.log_prior,
                torch.ones(fold.logits.shape[1], dtype=torch.float64),
            ).exp()[:, k]
            after = calibrated_log_probs(
                fold.logits, fold.log_prior, temperature
            ).exp()[:, k]
            for probs, conf_parts, correct_parts in (
                (before, before_conf, before_correct),
                (after, after_conf, after_correct),
            ):
                confidence, prediction = probs.max(-1)
                conf_parts.append(confidence)
                correct_parts.append(prediction.eq(fold.labels))
        before_stats = reliability_bins(
            torch.cat(before_conf), torch.cat(before_correct), n_bins
        )
        after_stats = reliability_bins(
            torch.cat(after_conf), torch.cat(after_correct), n_bins
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        for label, stats, offset in (
            ("Before", before_stats, -0.01),
            ("After", after_stats, 0.01),
        ):
            centers = (stats["lower"] + stats["upper"]) / 2 + offset
            mask = stats["count"] >= min_bin_count
            ax.plot(
                centers[mask], stats["accuracy"][mask], marker="o", label=label
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Confidence",
            ylabel="Empirical accuracy",
            title=f"Pooled reliability at k={k_one_based}",
        )
        ax.legend()
        fig.tight_layout()
        _atomic_save_figure(
            fig, figure_dir / f"reliability_k{k_one_based}.png"
        )
        plt.close(fig)


def calibration_history_mode(model_variant):
    """Return the unmoderated calibration history convention matching the training variant."""
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(
            f"unknown variant '{model_variant}', choices: {MODEL_VARIANTS}"
        )
    return VARIANTS[model_variant].calib_history


def _num_history_draws(history_mode, seeds):
    return 1 if history_mode == "full" else len(tuple(seeds))


def _checkpoint_variant(checkpoint):
    """When an older checkpoint lacks this field, default to V-Paired for backward compatibility."""
    return str(checkpoint.get("model_variant", "paired"))


def _require_checkpoint_variant(checkpoint, expected_variant, label):
    actual = _checkpoint_variant(checkpoint)
    if actual != expected_variant:
        raise ValueError(
            f"{label} model_variant {actual!r} != requested {expected_variant!r}"
        )


def dedup_and_stack(logits_draws, labels, occurrence_index):
    """Flatten [W,E,...] across history draws, keep only epochs with occurrence_index==0,
    and stack the S draws along the N dimension.

    logits_draws: sequence of S tensors, each [W, E, P, K]
    labels, occurrence_index: [W, E] (long)
    Returns (logits [n*S, P, K] float32, labels [n*S] long)
    """
    draws = list(logits_draws)
    if not draws:
        raise ValueError("logits_draws must not be empty")
    if labels.ndim != 2 or occurrence_index.shape != labels.shape:
        raise ValueError("labels and occurrence_index must both be [W, E]")
    windows, epochs = labels.shape
    keep = occurrence_index.reshape(-1) == 0
    kept_labels = labels.reshape(-1)[keep].long()
    logits_parts = []
    labels_parts = []
    for draw in draws:
        if draw.ndim != 4 or tuple(draw.shape[:2]) != (windows, epochs):
            raise ValueError("each draw must be [W, E, P, K] matching labels")
        flat = draw.reshape(windows * epochs, draw.shape[2], draw.shape[3])
        logits_parts.append(flat[keep].to(torch.float32))
        labels_parts.append(kept_labels)
    return torch.cat(logits_parts, dim=0), torch.cat(labels_parts, dim=0)


def generate_stacked_oof_logits(
    model,
    test_bundle,
    device,
    *,
    num_patches,
    num_epochs,
    seeds,
    batch_size,
    history_mode="random",
):
    """Generate deduplicated OOF logits ``[N,P,K]`` under full/random history."""
    seeds = tuple(int(s) for s in seeds)
    if history_mode not in ("full", "random"):
        raise ValueError("history_mode must be 'full' or 'random'")
    if history_mode == "random" and not seeds:
        raise ValueError("seeds must not be empty")
    num_windows = int(test_bundle.x.shape[0])
    loader = DataLoader(
        PauseWindowDataset(test_bundle),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    if history_mode == "full":
        history_inputs = ((None, None),)
    else:
        tables = build_random_history_tables(
            num_windows, num_epochs, num_patches, seeds=seeds
        )
        history_inputs = tuple((seed, tables[seed]) for seed in seeds)
    history_tables = tuple(table for _, table in history_inputs)
    logits_parts = [[] for _ in history_tables]
    label_parts = []
    occurrence_parts = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            # batch-first: prepare and upload x once, then compute all history draws sequentially.
            x = batch["x"].float().to(device, non_blocking=True)
            window_id = batch["window_id"].long()
            batch_draws = []
            for history_table in history_tables:
                if history_table is None:
                    end_idx = torch.full(
                        (x.shape[0], x.shape[1]),
                        num_patches - 1,
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    end_idx = history_table[window_id].to(device=device)
                batch_draws.append(model(x, end_idx))
            # Sync all draws for the current batch back to CPU at once, avoiding per-draw transfers.
            stacked = torch.stack(batch_draws, dim=0).cpu()
            for draw_index in range(len(history_tables)):
                logits_parts[draw_index].append(stacked[draw_index])
            label_parts.append(batch["y"].long())
            occurrence_parts.append(batch["occurrence_index"].long())

    logits_draws = [torch.cat(parts, dim=0) for parts in logits_parts]
    labels = torch.cat(label_parts, dim=0)
    occurrence_index = torch.cat(occurrence_parts, dim=0)
    return dedup_and_stack(logits_draws, labels, occurrence_index)


def discover_subject_paths(data_dir, subject_ids=SUBJECT_IDS):
    data_dir = Path(data_dir)
    paths = {}
    for subject_id in subject_ids:
        path = data_dir / f"subject{subject_id}.mat"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        paths[subject_id] = path
    return paths


def _log_prior_from_counts(class_counts):
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1 or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("class_counts must be a non-negative 1-D vector")
    return np.log(counts / counts.sum())


def write_oof_cache(
    output_path,
    *,
    subject_id,
    logits,
    labels,
    class_counts,
    train_mean,
    train_std,
    checkpoint_hash,
    data_fingerprint,
    model_config_json,
    seeds,
    model_variant="paired",
    history_mode="random",
):
    """Write variant-aware OOF logits cache for one fold."""
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 3 or labels.shape != (logits.shape[0],):
        raise ValueError("logits must be [N,P,K] and labels [N]")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN or Inf")
    atomic_savez(
        output_path,
        logits=logits,
        labels=labels,
        log_prior=_log_prior_from_counts(class_counts),
        class_counts=np.asarray(class_counts, dtype=np.int64),
        train_mean=np.asarray(train_mean, dtype=np.float64),
        train_std=np.asarray(train_std, dtype=np.float64),
        subject_id=np.asarray(subject_id, dtype=np.int64),
        checkpoint_hash=np.asarray(checkpoint_hash),
        data_fingerprint=np.asarray(data_fingerprint),
        model_config_json=np.asarray(model_config_json),
        model_variant=np.asarray(model_variant),
        history_mode=np.asarray(history_mode),
        history_end_idx_dist=np.asarray(history_end_idx_dist(history_mode)),
        history_seeds=np.asarray([int(s) for s in seeds], dtype=np.int64),
        num_history_draws=np.asarray(
            _num_history_draws(history_mode, seeds), dtype=np.int64
        ),
        occurrence_dedup=np.asarray(True),
    )


def build_pause_fold_cache(
    *,
    subject_id,
    subject_paths,
    checkpoint_path,
    output_path,
    device,
    num_epochs,
    num_classes,
    seeds,
    batch_size,
    data_fingerprint,
    model_variant="paired",
    history_mode="random",
):
    """Rebuild fold → validate variant → forward with specified history → deduplicate and write cache."""
    # Self-produced trusted checkpoint (contains Python containers), use weights_only=False.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _require_checkpoint_variant(ckpt, model_variant, f"subject{subject_id}")
    if int(ckpt["test_subject"]) != subject_id:
        raise ValueError(
            f"checkpoint test_subject {ckpt['test_subject']} != {subject_id}"
        )
    checkpoint_stats = np.asarray(
        [ckpt["train_mean"], ckpt["train_std"]], dtype=np.float64
    )
    if not np.isfinite(checkpoint_stats).all() or checkpoint_stats[1] <= 0:
        raise ValueError("checkpoint normalization stats must be finite and positive")
    ordered = sorted(subject_paths)
    subject_index = {path: sid for sid, path in subject_paths.items()}
    fold_spec = Fold(
        fold_id=subject_id,
        train_paths=[subject_paths[sid] for sid in ordered if sid != subject_id],
        test_paths=[subject_paths[subject_id]],
        test_units=[subject_id],
        subject_index=subject_index,
    )
    fold = assemble_calibration_fold(
        fold_spec,
        train_mean=checkpoint_stats[0],
        train_std=checkpoint_stats[1],
        num_epochs=num_epochs,
        num_classes=num_classes,
    )
    model_kwargs = dict(ckpt["model_kwargs"])
    num_patches = int(model_kwargs["num_patches"])
    if int(model_kwargs["num_classes"]) != num_classes:
        raise ValueError("checkpoint num_classes mismatch")
    model = PAUSE(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    logits, labels = generate_stacked_oof_logits(
        model, fold.test, device,
        num_patches=num_patches, num_epochs=num_epochs,
        seeds=seeds, batch_size=batch_size, history_mode=history_mode,
    )
    logits = logits.numpy().astype(np.float32, copy=False)
    labels = labels.numpy().astype(np.int64, copy=False)
    if logits.shape[1:] != (num_patches, num_classes):
        raise ValueError(f"unexpected logits shape {logits.shape}")

    write_oof_cache(
        output_path,
        subject_id=subject_id,
        logits=logits,
        labels=labels,
        class_counts=fold.class_counts,
        train_mean=fold.train_mean,
        train_std=fold.train_std,
        checkpoint_hash=sha256_file(checkpoint_path),
        data_fingerprint=data_fingerprint,
        model_config_json=_model_config_json(model_kwargs, num_classes),
        seeds=seeds,
        model_variant=model_variant,
        history_mode=history_mode,
    )


def pause_cache_matches_contract(
    path, *, expected_model_config_json, expected_seeds,
    expected_model_variant="paired", expected_history_mode="random",
):
    """Validate PAUSE variant, history condition, and model config against the cache contract."""
    expected_seeds = [int(seed) for seed in expected_seeds]
    try:
        with np.load(path, allow_pickle=False) as saved:
            required = {
                "model_config_json", "history_seeds", "num_history_draws",
                "occurrence_dedup", "model_variant", "history_mode",
                "history_end_idx_dist",
            }
            if not required.issubset(saved.files):
                return False
            return (
                str(saved["model_config_json"].item())
                == expected_model_config_json
                and str(saved["model_variant"].item())
                == expected_model_variant
                and str(saved["history_mode"].item())
                == expected_history_mode
                and str(saved["history_end_idx_dist"].item())
                == history_end_idx_dist(expected_history_mode)
                and saved["history_seeds"].tolist() == expected_seeds
                and int(saved["num_history_draws"].item())
                == _num_history_draws(expected_history_mode, expected_seeds)
                and bool(saved["occurrence_dedup"].item())
            )
    except (OSError, ValueError, KeyError):
        return False


def validate_pause_cache_fingerprints(
    *,
    data_dir,
    checkpoint_dir,
    cache_dir,
    subject_ids=SUBJECT_IDS,
    num_epochs=NUM_EPOCHS,
    num_classes=NUM_CLASSES,
    seeds,
    device="cpu",
    model_variant="paired",
    history_mode="random",
):
    """Prevent the fit-only path from fitting an old cache and then mislabeling it with a new manifest."""
    subject_paths = discover_subject_paths(data_dir, subject_ids)
    data_fingerprint = fingerprint_files(subject_paths)
    for subject_id in subject_ids:
        checkpoint = Path(checkpoint_dir) / f"subject{subject_id}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        _require_checkpoint_variant(ckpt, model_variant, f"subject{subject_id}")
        model_kwargs = dict(ckpt["model_kwargs"])
        if (
            int(model_kwargs["num_epochs"]) != num_epochs
            or int(model_kwargs["num_classes"]) != num_classes
        ):
            raise ValueError(f"subject{subject_id} checkpoint model contract mismatch")
        cache_path = Path(cache_dir) / f"subject{subject_id}.npz"
        load_fold_cache(
            cache_path,
            expected_subject_id=subject_id,
            expected_checkpoint_hash=sha256_file(checkpoint),
            expected_data_fingerprint=data_fingerprint,
        )
        expected_config = _model_config_json(model_kwargs, num_classes)
        if not pause_cache_matches_contract(
            cache_path,
            expected_model_config_json=expected_config,
            expected_seeds=seeds,
            expected_model_variant=model_variant,
            expected_history_mode=history_mode,
        ):
            raise ValueError(f"subject{subject_id} cache model/history seeds contract mismatch")


def run_pause_cache_stage(
    *,
    data_dir,
    checkpoint_dir,
    cache_dir,
    subject_ids=SUBJECT_IDS,
    num_epochs=NUM_EPOCHS,
    num_classes=NUM_CLASSES,
    seeds,
    batch_size=BATCH_SIZE_WINDOWS,
    device="cpu",
    force=False,
    model_variant="paired",
    history_mode="random",
):
    subject_paths = discover_subject_paths(data_dir, subject_ids)
    data_fingerprint = fingerprint_files(subject_paths)
    cache_dir = Path(cache_dir)
    seeds = tuple(int(s) for s in seeds)
    for subject_id in subject_ids:
        checkpoint = Path(checkpoint_dir) / f"subject{subject_id}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing {checkpoint}")
        output = cache_dir / f"subject{subject_id}.npz"
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        _require_checkpoint_variant(ckpt, model_variant, f"subject{subject_id}")
        model_kwargs = dict(ckpt["model_kwargs"])
        if (
            int(model_kwargs["num_epochs"]) != num_epochs
            or int(model_kwargs["num_classes"]) != num_classes
        ):
            raise ValueError(f"subject{subject_id} checkpoint model contract mismatch")
        expected_config = _model_config_json(model_kwargs, num_classes)
        if output.is_file() and not force:
            try:
                load_fold_cache(
                    output,
                    expected_subject_id=subject_id,
                    expected_checkpoint_hash=sha256_file(checkpoint),
                    expected_data_fingerprint=data_fingerprint,
                )
                if pause_cache_matches_contract(
                    output,
                    expected_model_config_json=expected_config,
                    expected_seeds=seeds,
                    expected_model_variant=model_variant,
                    expected_history_mode=history_mode,
                ):
                    LOGGER.info("subject%s cache reused", subject_id)
                    continue
            except (ValueError, KeyError):
                pass
        LOGGER.info("subject%s building cache", subject_id)
        build_pause_fold_cache(
            subject_id=subject_id,
            subject_paths=subject_paths,
            checkpoint_path=checkpoint,
            output_path=output,
            device=device,
            num_epochs=num_epochs,
            num_classes=num_classes,
            seeds=seeds,
            batch_size=batch_size,
            data_fingerprint=data_fingerprint,
            model_variant=model_variant,
            history_mode=history_mode,
        )


def build_pause_manifest(
    *, subject_paths, checkpoint_paths, seeds, n_bins, device,
    model_variant="paired", history_mode="random", data_hashes=None,
    checkpoint_hashes=None,
):
    seeds = tuple(int(s) for s in seeds)
    if history_mode not in ("full", "random"):
        raise ValueError("history_mode must be 'full' or 'random'")
    return {
        "protocol": f"cross-fitted-per-k-temperature-{history_mode}-history",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "subjects": sorted(subject_paths),
        "subject_weighting": "equal",
        "model_variant": model_variant,
        "history_mode": history_mode,
        "history_end_idx_dist": history_end_idx_dist(history_mode),
        "num_history_draws": _num_history_draws(history_mode, seeds),
        "history_seeds": list(seeds),
        "occurrence_dedup": True,
        "num_epochs": NUM_EPOCHS,
        "temperature_parameterization": "T=exp(log_T)",
        "optimizer": {
            "name": "torch.optim.LBFGS",
            "lr": 1.0,
            "max_iter": 50,
            "line_search_fn": "strong_wolfe",
            "fallback": "scipy.optimize.minimize_scalar bounded in log(T)",
            "bounds": [0.05, 20.0],
        },
        "ece_bins": n_bins,
        "device": device,
        "num_classes": NUM_CLASSES,
        "data_sha256": {
            str(sid): (
                data_hashes[Path(path)]
                if data_hashes is not None
                else sha256_file(path)
            )
            for sid, path in sorted(subject_paths.items())
        },
        "checkpoint_sha256": {
            str(sid): (
                checkpoint_hashes[sid]
                if checkpoint_hashes is not None
                else sha256_file(path)
            )
            for sid, path in sorted(checkpoint_paths.items())
        },
    }


def _fold_stem(cfg, fold_id):
    return cache_stem_fmt(cfg).format(fold_id=fold_id)


def _hash_paths(paths):
    """Compute SHA256 once for each deduplicated path."""
    unique = sorted({Path(path) for path in paths}, key=lambda p: str(p))
    return {path: sha256_file(path) for path in unique}


def _fingerprint_from_hashes(path_hashes):
    """Generate an aggregate dataset fingerprint from pre-computed per-file SHA256 hashes."""
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(path_hashes, key=lambda p: str(p)):
        digest.update(path.name.encode("utf-8"))
        digest.update(path_hashes[path].encode("ascii"))
    return digest.hexdigest()


def _fingerprint_paths(paths):
    """Convenience entry point: hash each file once, then produce the aggregate fingerprint."""
    return _fingerprint_from_hashes(_hash_paths(paths))


def build_pause_fold_cache_for_fold(
    *, fold, checkpoint_path, output_path, device, num_epochs, num_classes,
    seeds, batch_size, data_fingerprint, model_variant="paired",
    history_mode="random", checkpoint_hash=None,
):
    """Generic fold cache: the subject_id field inside the checkpoint/cache carries the fold_id."""
    unit_id = fold.fold_id
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _require_checkpoint_variant(ckpt, model_variant, f"fold{unit_id}")
    if int(ckpt["fold_id"]) != unit_id:
        raise ValueError(f"checkpoint fold_id {ckpt['fold_id']} != {unit_id}")
    checkpoint_stats = np.asarray([ckpt["train_mean"], ckpt["train_std"]], dtype=np.float64)
    if not np.isfinite(checkpoint_stats).all() or checkpoint_stats[1] <= 0:
        raise ValueError("checkpoint normalization stats must be finite and positive")
    pause_fold = assemble_calibration_fold(
        fold,
        train_mean=checkpoint_stats[0],
        train_std=checkpoint_stats[1],
        num_epochs=num_epochs,
        num_classes=num_classes,
    )
    model_kwargs = dict(ckpt["model_kwargs"])
    num_patches = int(model_kwargs["num_patches"])
    if int(model_kwargs["num_classes"]) != num_classes:
        raise ValueError("checkpoint num_classes mismatch")
    model = PAUSE(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    logits, labels = generate_stacked_oof_logits(
        model, pause_fold.test, device, num_patches=num_patches,
        num_epochs=num_epochs, seeds=seeds, batch_size=batch_size,
        history_mode=history_mode,
    )
    logits = logits.numpy().astype(np.float32, copy=False)
    labels = labels.numpy().astype(np.int64, copy=False)
    if logits.shape[1:] != (num_patches, num_classes):
        raise ValueError(f"unexpected logits shape {logits.shape}")
    write_oof_cache(
        output_path, subject_id=unit_id, logits=logits, labels=labels,
        class_counts=pause_fold.class_counts, train_mean=pause_fold.train_mean,
        train_std=pause_fold.train_std,
        checkpoint_hash=checkpoint_hash or sha256_file(checkpoint_path),
        data_fingerprint=data_fingerprint,
        model_config_json=_model_config_json(model_kwargs, num_classes), seeds=seeds,
        model_variant=model_variant, history_mode=history_mode,
    )


def _validate_fold_checkpoint(
    checkpoint, *, fold_id, num_epochs, num_classes, model_variant="paired"
):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require_checkpoint_variant(ckpt, model_variant, f"fold{fold_id}")
    kwargs = dict(ckpt["model_kwargs"])
    if (int(ckpt["fold_id"]) != fold_id
            or int(kwargs["num_epochs"]) != num_epochs
            or int(kwargs["num_classes"]) != num_classes):
        raise ValueError(f"fold{fold_id} checkpoint model contract mismatch")
    return kwargs


def run_pause_cache_stage_for_cfg(
    *, cfg, folds, checkpoint_dir, cache_dir, seeds, batch_size, num_epochs,
    num_classes, device="cpu", force=False, model_variant="paired",
    history_mode="random", data_fingerprint=None, checkpoint_hashes=None,
):
    if data_fingerprint is None:
        all_paths = [path for fold in folds for path in fold.subject_index]
        data_fingerprint = _fingerprint_paths(all_paths)
    cache_dir = Path(cache_dir)
    for fold in folds:
        stem = _fold_stem(cfg, fold.fold_id)
        checkpoint = Path(checkpoint_dir) / cfg.ckpt_name_fmt.format(fold_id=fold.fold_id)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing {checkpoint}")
        checkpoint_hash = (
            checkpoint_hashes[fold.fold_id]
            if checkpoint_hashes is not None
            else sha256_file(checkpoint)
        )
        output = cache_dir / f"{stem}.npz"
        kwargs = _validate_fold_checkpoint(
            checkpoint, fold_id=fold.fold_id, num_epochs=num_epochs,
            num_classes=num_classes, model_variant=model_variant,
        )
        expected_config = _model_config_json(kwargs, num_classes)
        if output.is_file() and not force:
            try:
                load_fold_cache(output, expected_subject_id=fold.fold_id,
                                expected_checkpoint_hash=checkpoint_hash,
                                expected_data_fingerprint=data_fingerprint)
                if pause_cache_matches_contract(output, expected_model_config_json=expected_config,
                                                expected_seeds=seeds,
                                                expected_model_variant=model_variant,
                                                expected_history_mode=history_mode):
                    LOGGER.info("%s cache reused", stem)
                    continue
            except (ValueError, KeyError):
                pass
        LOGGER.info("%s building cache", stem)
        build_pause_fold_cache_for_fold(
            fold=fold, checkpoint_path=checkpoint, output_path=output, device=device,
            num_epochs=num_epochs, num_classes=num_classes, seeds=seeds,
            batch_size=batch_size, data_fingerprint=data_fingerprint,
            model_variant=model_variant, history_mode=history_mode,
            checkpoint_hash=checkpoint_hash,
        )


def validate_pause_cache_fingerprints_for_cfg(
    *, cfg, folds, checkpoint_dir, cache_dir, seeds, num_epochs, num_classes,
    model_variant="paired", history_mode="random", data_fingerprint=None,
    checkpoint_hashes=None,
):
    if data_fingerprint is None:
        data_fingerprint = _fingerprint_paths(
            path for fold in folds for path in fold.subject_index
        )
    for fold in folds:
        stem = _fold_stem(cfg, fold.fold_id)
        checkpoint = Path(checkpoint_dir) / cfg.ckpt_name_fmt.format(fold_id=fold.fold_id)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing {checkpoint}")
        kwargs = _validate_fold_checkpoint(
            checkpoint, fold_id=fold.fold_id, num_epochs=num_epochs,
            num_classes=num_classes, model_variant=model_variant,
        )
        cache_path = Path(cache_dir) / f"{stem}.npz"
        checkpoint_hash = (
            checkpoint_hashes[fold.fold_id]
            if checkpoint_hashes is not None
            else sha256_file(checkpoint)
        )
        load_fold_cache(cache_path, expected_subject_id=fold.fold_id,
                        expected_checkpoint_hash=checkpoint_hash,
                        expected_data_fingerprint=data_fingerprint)
        if not pause_cache_matches_contract(
            cache_path, expected_model_config_json=_model_config_json(kwargs, num_classes),
            expected_seeds=seeds,
            expected_model_variant=model_variant,
            expected_history_mode=history_mode,
        ):
            raise ValueError(f"{stem} cache model/history seeds contract mismatch")


def build_pause_manifest_for_cfg(
    *, cfg, folds, checkpoint_dir, seeds, n_bins, device,
    model_variant="paired", history_mode="random", data_hashes=None,
    checkpoint_hashes=None,
):
    paths = sorted({path for fold in folds for path in fold.subject_index}, key=lambda p: str(p))
    checkpoints = {
        fold.fold_id: Path(checkpoint_dir) / cfg.ckpt_name_fmt.format(fold_id=fold.fold_id)
        for fold in folds
    }
    manifest = build_pause_manifest(
        subject_paths={path.name: path for path in paths}, checkpoint_paths=checkpoints,
        seeds=seeds, n_bins=n_bins, device=device, model_variant=model_variant,
        history_mode=history_mode, data_hashes=data_hashes,
        checkpoint_hashes=checkpoint_hashes,
    )
    manifest.update(dataset=cfg.name, fold_ids=[fold.fold_id for fold in folds])
    return manifest


def run_calibration(
    cfg, *, checkpoint_dir=None, output_dir=None, stage="all", batch_size=None,
    n_bins=15, seed_start=DEFAULT_SEED_START, num_history_draws=DEFAULT_NUM_HISTORY_DRAWS,
    device="cpu", force_cache=False, variant="paired",
):
    history_mode = calibration_history_mode(variant)
    if num_history_draws < 1:
        raise ValueError("num_history_draws must be positive")
    checkpoint_dir = Path(checkpoint_dir or cfg_ckpt_dir(cfg, variant))
    output = Path(output_dir or cfg_calib_dir(cfg, variant))
    batch_size = batch_size or cfg.batch_size_windows
    seeds = (
        tuple(range(seed_start, seed_start + num_history_draws))
        if history_mode == "random"
        else ()
    )
    folds = enumerate_folds(cfg)
    fold_ids = [fold.fold_id for fold in folds]
    data_paths = sorted(
        {path for fold in folds for path in fold.subject_index},
        key=lambda path: str(path),
    )
    checkpoint_paths = {
        fold.fold_id: checkpoint_dir / cfg.ckpt_name_fmt.format(fold_id=fold.fold_id)
        for fold in folds
    }
    # Share a single hash snapshot across the run; cache/fit validation and manifest avoid re-reading files.
    data_hashes = _hash_paths(data_paths)
    data_fingerprint = _fingerprint_from_hashes(data_hashes)
    checkpoint_hashes = {
        fold_id: sha256_file(path) for fold_id, path in checkpoint_paths.items()
    }

    def stem_for_id(fold_id):
        return _fold_stem(cfg, fold_id)

    cache_dir, temperature_dir, metrics_dir = (
        output / "oof_logits", output / "temperatures", output / "metrics"
    )
    if stage in ("cache", "all"):
        run_pause_cache_stage_for_cfg(
            cfg=cfg, folds=folds, checkpoint_dir=checkpoint_dir, cache_dir=cache_dir,
            seeds=seeds, batch_size=batch_size, num_epochs=NUM_EPOCHS,
            num_classes=NUM_CLASSES, device=device, force=force_cache,
            model_variant=variant, history_mode=history_mode,
            data_fingerprint=data_fingerprint,
            checkpoint_hashes=checkpoint_hashes,
        )
    if stage in ("fit", "all"):
        validate_pause_cache_fingerprints_for_cfg(
            cfg=cfg, folds=folds, checkpoint_dir=checkpoint_dir, cache_dir=cache_dir,
            seeds=seeds, num_epochs=NUM_EPOCHS, num_classes=NUM_CLASSES,
            model_variant=variant, history_mode=history_mode,
            data_fingerprint=data_fingerprint,
            checkpoint_hashes=checkpoint_hashes,
        )
        run_fit_stage(cache_dir=cache_dir, temperature_dir=temperature_dir,
                      metrics_dir=metrics_dir, subject_ids=fold_ids, n_bins=n_bins,
                      stem_for_id=stem_for_id)
        validate_complete_run(temperature_dir=temperature_dir, metrics_dir=metrics_dir,
                              subject_ids=fold_ids, stem_for_id=stem_for_id)
        write_summary_csv(metrics_dir, output / "summary.csv", subject_ids=fold_ids,
                          stem_for_id=stem_for_id)
        write_temperature_curve_csv(temperature_dir, output / "temperature_curve.csv",
                                    subject_ids=fold_ids, stem_for_id=stem_for_id)
        write_figures(cache_dir=cache_dir, temperature_dir=temperature_dir,
                      metrics_dir=metrics_dir, figure_dir=output / "figures",
                      subject_ids=fold_ids, n_bins=n_bins, stem_for_id=stem_for_id)
        atomic_write_text(output / "manifest.json", json.dumps(
            build_pause_manifest_for_cfg(cfg=cfg, folds=folds,
                                         checkpoint_dir=checkpoint_dir, seeds=seeds,
                                         n_bins=n_bins, device=device,
                                         model_variant=variant,
                                         history_mode=history_mode,
                                         data_hashes=data_hashes,
                                         checkpoint_hashes=checkpoint_hashes),
            ensure_ascii=False, indent=2,
        ) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="PAUSE cross-epoch, variant-aware per-k temperature calibration"
    )
    parser.add_argument("--dataset", required=True,
                        choices=("dreams", "edf20", "edf78", "shhs"))
    parser.add_argument("--stage", choices=("cache", "fit", "all"), default="all")
    parser.add_argument(
        "--variant", choices=MODEL_VARIANTS, default="paired",
        help="Training variant; history convention determined by VARIANTS: full/nohist=complete, paired/random=random",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument(
        "--num-history-draws", type=int, default=DEFAULT_NUM_HISTORY_DRAWS
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    run_calibration(
        get_cfg(args.dataset), checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir, stage=args.stage, batch_size=args.batch_size,
        n_bins=args.n_bins, seed_start=args.seed_start,
        num_history_draws=args.num_history_draws, device=args.device,
        force_cache=args.force_cache, variant=args.variant,
    )


if __name__ == "__main__":
    main()
