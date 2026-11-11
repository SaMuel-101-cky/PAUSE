import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from entry.gp.gp_identity import identity_for_fold, resolve_read_artifact
from models.gp.contracts import GPLogitShard, LoadedTemperature


_OOF_REQUIRED_KEYS = {
    "logits",
    "labels",
    "log_prior",
    "class_counts",
    "train_mean",
    "train_std",
    "checkpoint_hash",
    "data_fingerprint",
    "model_config_json",
    "history_seeds",
    "num_history_draws",
    "occurrence_dedup",
}
_PAUSE_NUM_EPOCHS = 32


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generic_data_fingerprint(paths) -> str:
    digest = hashlib.sha256()
    unique_paths = sorted({Path(path) for path in paths}, key=lambda path: str(path))
    if not unique_paths:
        raise ValueError("data fingerprint requires at least one path")
    for path in unique_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing data file: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def dreams_legacy_data_fingerprint(subject_index) -> str:
    if not subject_index:
        raise ValueError("DREAMS data fingerprint requires subject_index")
    by_subject = {}
    for path, subject_id in subject_index.items():
        subject_id = int(subject_id)
        path = Path(path)
        previous = by_subject.setdefault(subject_id, path)
        if previous != path:
            raise ValueError(
                f"DREAMS subject {subject_id} maps to multiple data files"
            )
    digest = hashlib.sha256()
    for subject_id, path in sorted(by_subject.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing data file: {path}")
        digest.update(str(subject_id).encode("utf-8"))
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _require_keys(saved, required, *, artifact):
    missing = set(required).difference(saved.files)
    if missing:
        raise ValueError(f"{artifact} missing keys: {sorted(missing)}")


def _artifact_scalar(saved, key, artifact):
    if key not in saved.files:
        raise ValueError(f"{artifact} missing {key}")
    value = np.asarray(saved[key])
    if value.shape != ():
        raise ValueError(f"{artifact} {key} must be scalar")
    return str(value.item())


def _validate_optional_artifact_scalar(saved, key, expected, artifact):
    if key not in saved.files:
        return
    actual = _artifact_scalar(saved, key, artifact)
    if actual != expected:
        raise ValueError(
            f"{artifact} {key} does not match expected "
            f"{expected!r}: {actual!r}"
        )


def _canonical_id_from_oof(saved, dataset: str) -> int:
    candidates = []
    if "fold_id" in saved.files:
        candidates.append(int(saved["fold_id"].item()))
    if "subject_id" in saved.files:
        subject_id = int(saved["subject_id"].item())
        candidates.append(subject_id - 1 if dataset == "dreams" else subject_id)
    if not candidates:
        raise ValueError("OOF cache missing identity field")
    if len(set(candidates)) != 1:
        raise ValueError(f"OOF cache has conflicting identity metadata: {candidates}")
    return candidates[0]


def _model_dimensions(
    model_config_json: str,
) -> tuple[int | None, int | None, int | None]:
    try:
        config = json.loads(model_config_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("model_config_json is not valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("model_config_json must contain an object")
    model_kwargs = config.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise ValueError("model_config_json model_kwargs must be an object")
    epochs = model_kwargs.get("num_epochs", config.get("num_epochs"))
    patches = model_kwargs.get("num_patches", config.get("num_patches"))
    classes = config.get("num_classes", model_kwargs.get("num_classes"))
    return (
        None if epochs is None else int(epochs),
        None if patches is None else int(patches),
        None if classes is None else int(classes),
    )


def load_oof_shard_file(
    path,
    *,
    identity,
    dataset,
    history_draws,
    expected_history_seeds,
    max_history_draws,
    expected_model_variant,
    expected_history_mode,
    expected_patches,
    expected_classes,
    accepted_checkpoint_hashes,
    accepted_data_fingerprints,
) -> GPLogitShard:
    path = Path(path)
    expected_history_seeds = tuple(int(seed) for seed in expected_history_seeds)
    history_draws = int(history_draws)
    max_history_draws = int(max_history_draws)
    if not 1 <= history_draws <= max_history_draws:
        raise ValueError(
            f"history_draws must be in [1, {max_history_draws}]"
        )
    if expected_history_mode not in ("full", "random"):
        raise ValueError("expected_history_mode must be 'full' or 'random'")

    with np.load(path, allow_pickle=False) as saved:
        _require_keys(saved, _OOF_REQUIRED_KEYS, artifact="OOF cache")
        _validate_optional_artifact_scalar(
            saved, "model_variant", expected_model_variant, "OOF cache"
        )
        _validate_optional_artifact_scalar(
            saved, "history_mode", expected_history_mode, "OOF cache"
        )
        canonical_id = _canonical_id_from_oof(saved, dataset)
        if canonical_id != identity.canonical_fold_id:
            raise ValueError(
                f"OOF identity {canonical_id} != expected "
                f"{identity.canonical_fold_id}"
            )

        logits = np.asarray(saved["logits"])
        labels = np.asarray(saved["labels"])
        log_prior = np.asarray(saved["log_prior"])
        class_counts = np.asarray(saved["class_counts"])
        if logits.ndim != 3 or logits.shape[1:] != (
            expected_patches,
            expected_classes,
        ):
            raise ValueError(
                "OOF logits shape must be "
                f"[N,{expected_patches},{expected_classes}], got {logits.shape}"
            )
        if not np.isfinite(logits).all():
            raise ValueError("OOF logits must be finite")
        if labels.shape != (logits.shape[0],):
            raise ValueError("OOF labels must match logits rows")
        if class_counts.shape != (expected_classes,):
            raise ValueError("OOF class_counts shape mismatch")
        if np.any(class_counts <= 0):
            raise ValueError("OOF class_counts must be positive")
        expected_prior = np.log(class_counts / class_counts.sum())
        if log_prior.shape != (expected_classes,) or not np.allclose(
            log_prior, expected_prior, atol=1e-8, rtol=0.0
        ):
            raise ValueError("OOF log_prior does not match class_counts")

        train_mean = float(saved["train_mean"].item())
        train_std = float(saved["train_std"].item())
        if not np.isfinite([train_mean, train_std]).all() or train_std <= 0:
            raise ValueError("OOF normalization statistics are invalid")

        checkpoint_hash = str(saved["checkpoint_hash"].item())
        if checkpoint_hash not in set(accepted_checkpoint_hashes):
            raise ValueError("OOF checkpoint hash does not match current checkpoint")
        data_fingerprint = str(saved["data_fingerprint"].item())
        if data_fingerprint not in set(accepted_data_fingerprints):
            raise ValueError("OOF data fingerprint does not match current data")
        if not bool(saved["occurrence_dedup"].item()):
            raise ValueError("OOF occurrence_dedup must be true")

        stored_draws = int(saved["num_history_draws"].item())
        stored_seeds = tuple(
            int(seed) for seed in np.asarray(saved["history_seeds"]).tolist()
        )
        if stored_draws != max_history_draws:
            raise ValueError("OOF num_history_draws does not match expected draws")
        if expected_history_mode == "full":
            if expected_history_seeds or stored_seeds or stored_draws != 1:
                raise ValueError(
                    "full-history OOF requires one draw and an empty seed list"
                )
        elif stored_seeds != expected_history_seeds:
            raise ValueError("OOF history seeds do not match expected seeds")
        if logits.shape[0] % stored_draws != 0:
            raise ValueError("OOF rows are not divisible by num_history_draws")

        model_epochs, model_patches, model_classes = _model_dimensions(
            str(saved["model_config_json"].item())
        )
        if (
            model_epochs != _PAUSE_NUM_EPOCHS
            or model_patches != expected_patches
            or model_classes != expected_classes
        ):
            raise ValueError("OOF model config does not match expected E/P/K")

        epochs_per_draw = logits.shape[0] // stored_draws
        reference_labels = labels[:epochs_per_draw]
        for draw in range(1, stored_draws):
            start = draw * epochs_per_draw
            stop = start + epochs_per_draw
            if not np.array_equal(labels[start:stop], reference_labels):
                raise ValueError("OOF label blocks differ across history draws")
        rows = history_draws * epochs_per_draw
        return GPLogitShard(
            canonical_fold_id=canonical_id,
            logits=logits[:rows].copy(),
            labels=labels[:rows].copy(),
            log_prior=log_prior.copy(),
            oof_path=str(path),
            oof_sha256=sha256_file(path),
            checkpoint_sha256=checkpoint_hash,
            data_fingerprint=data_fingerprint,
        )


def _canonical_temperature_ids(saved, dataset: str, singular: bool):
    fold_key = "target_fold" if singular else "source_folds"
    subject_key = "target_subject" if singular else "source_subjects"
    candidates = []
    if fold_key in saved.files:
        candidates.append(np.asarray(saved[fold_key], dtype=np.int64))
    if subject_key in saved.files:
        subject_ids = np.asarray(saved[subject_key], dtype=np.int64)
        candidates.append(subject_ids - 1 if dataset == "dreams" else subject_ids)
    if not candidates:
        raise ValueError(
            f"temperature missing {'target' if singular else 'source'} identity"
        )
    normalized = [tuple(array.reshape(-1).tolist()) for array in candidates]
    if len(set(normalized)) != 1:
        raise ValueError("temperature has conflicting identity metadata")
    if singular:
        if len(normalized[0]) != 1:
            raise ValueError("temperature target identity must be scalar")
        return normalized[0][0]
    return normalized[0]


def load_temperature_file(
    path,
    *,
    dataset,
    target_identity,
    expected_source_ids,
    expected_patches,
    expected_model_variant,
    expected_history_mode,
) -> LoadedTemperature:
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        _require_keys(saved, {"temperature"}, artifact="temperature")
        _validate_optional_artifact_scalar(
            saved, "model_variant", expected_model_variant, "temperature"
        )
        _validate_optional_artifact_scalar(
            saved, "history_mode", expected_history_mode, "temperature"
        )
        target_id = _canonical_temperature_ids(saved, dataset, singular=True)
        if target_id != target_identity.canonical_fold_id:
            raise ValueError(
                f"temperature target identity {target_id} != expected "
                f"{target_identity.canonical_fold_id}"
            )
        source_ids = _canonical_temperature_ids(saved, dataset, singular=False)
        expected = tuple(sorted(int(fold_id) for fold_id in expected_source_ids))
        if len(source_ids) != len(set(source_ids)) or tuple(sorted(source_ids)) != expected:
            raise ValueError(
                f"temperature source folds {sorted(source_ids)} != expected "
                f"{list(expected)}"
            )

        temperature = np.asarray(saved["temperature"], dtype=np.float32)
        if temperature.shape != (expected_patches,):
            raise ValueError(
                f"temperature shape must be [{expected_patches}], got "
                f"{temperature.shape}"
            )
        if not np.isfinite(temperature).all():
            raise ValueError("temperature values must be finite")
        if not np.all(temperature > 0):
            raise ValueError("temperature values must be positive")

    return LoadedTemperature(
        temperature=torch.from_numpy(temperature.copy()),
        source_fold_ids=expected,
        path=str(path),
        sha256=sha256_file(path),
    )


def _indexed_fold_identities(profile, all_folds):
    identities = {}
    for fold in all_folds:
        identity = identity_for_fold(profile.dataset, fold)
        if identity.canonical_fold_id in identities:
            raise ValueError(
                f"duplicate canonical fold id {identity.canonical_fold_id}"
            )
        identities[identity.canonical_fold_id] = (fold, identity)
    expected = set(range(profile.pause_cfg.num_folds))
    if set(identities) != expected:
        raise ValueError(
            f"PAUSE OOF folds {sorted(identities)} != expected {sorted(expected)}"
        )
    return identities


def _combined_subject_index(all_folds):
    combined = {}
    for fold in all_folds:
        for path, subject_id in fold.subject_index.items():
            path = Path(path)
            subject_id = int(subject_id)
            previous = combined.setdefault(path, subject_id)
            if previous != subject_id:
                raise ValueError(
                    f"data path {path} has conflicting subject ids "
                    f"{previous} and {subject_id}"
                )
    return combined


def _expected_patches(profile) -> int:
    return int(profile.model_kwargs.get("num_patches", 30))


def load_pause_training_shards(
    profile,
    *,
    target_fold,
    all_folds,
    history_draws,
) -> tuple[GPLogitShard, ...]:
    if profile.oof_logits_dir is None:
        raise ValueError("PAUSE profile must declare oof_logits_dir")
    all_folds = tuple(all_folds)
    indexed = _indexed_fold_identities(profile, all_folds)
    target_identity = identity_for_fold(profile.dataset, target_fold)
    if target_identity.canonical_fold_id not in indexed:
        raise ValueError("target fold is not present in all_folds")
    source_ids = sorted(set(indexed) - {target_identity.canonical_fold_id})

    cache_paths = {}
    checkpoint_paths = {}
    missing = []
    for source_id in source_ids:
        _, identity = indexed[source_id]
        try:
            cache_paths[source_id] = resolve_read_artifact(
                profile.oof_logits_dir,
                identity,
                ".npz",
            )
        except FileNotFoundError as exc:
            missing.append(str(exc))
        try:
            checkpoint_paths[source_id] = resolve_read_artifact(
                profile.s1_checkpoint_dir,
                identity,
                ".pth",
            )
        except FileNotFoundError as exc:
            missing.append(str(exc))
    if missing:
        raise FileNotFoundError(
            "missing PAUSE OOF source artifacts:\n- " + "\n- ".join(missing)
        )

    subject_index = _combined_subject_index(all_folds)
    fingerprints = {generic_data_fingerprint(subject_index)}
    if profile.dataset == "dreams":
        fingerprints.add(dreams_legacy_data_fingerprint(subject_index))
    shards = []
    for source_id in source_ids:
        checkpoint_hash = sha256_file(checkpoint_paths[source_id])
        shard = load_oof_shard_file(
            cache_paths[source_id],
            identity=indexed[source_id][1],
            dataset=profile.dataset,
            history_draws=history_draws,
            expected_history_seeds=profile.history_seeds,
            max_history_draws=profile.num_history_draws,
            expected_model_variant=profile.model_variant,
            expected_history_mode=profile.calibration_history_mode,
            expected_patches=_expected_patches(profile),
            expected_classes=profile.num_classes,
            accepted_checkpoint_hashes={checkpoint_hash},
            accepted_data_fingerprints=fingerprints,
        )
        shards.append(
            replace(
                shard,
                checkpoint_path=str(checkpoint_paths[source_id]),
                checkpoint_sha256=checkpoint_hash,
            )
        )
    return tuple(shards)


def load_pause_target_temperature(profile, *, fold, all_folds) -> LoadedTemperature:
    if profile.temperature_dir is None:
        raise ValueError("PAUSE profile must declare temperature_dir")
    all_folds = tuple(all_folds)
    indexed = _indexed_fold_identities(profile, all_folds)
    target_identity = identity_for_fold(profile.dataset, fold)
    if target_identity.canonical_fold_id not in indexed:
        raise ValueError("target fold is not present in all_folds")
    source_ids = tuple(
        sorted(set(indexed) - {target_identity.canonical_fold_id})
    )
    temperature_path = resolve_read_artifact(
        profile.temperature_dir,
        target_identity,
        ".npz",
    )
    return load_temperature_file(
        temperature_path,
        dataset=profile.dataset,
        target_identity=target_identity,
        expected_source_ids=source_ids,
        expected_patches=_expected_patches(profile),
        expected_model_variant=profile.model_variant,
        expected_history_mode=profile.calibration_history_mode,
    )
