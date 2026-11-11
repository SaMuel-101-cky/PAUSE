import json
from collections.abc import Mapping
from pathlib import Path
import tempfile

import numpy as np
import torch

from dataset.pause import assemble_fold
from entry.gp.gp_helpers import load_checkpoint_payload
from entry.gp.gp_identity import identity_for_fold
from entry.gp.pause_oof import (
    load_pause_target_temperature,
    load_pause_training_shards,
    sha256_file,
)
from models.gp.decision import decision_helper
from models.gp.llr_utils import extract_llrs_from_logits
from models.pause.model import PAUSE


def _early_stopping_summary(row):
    if not isinstance(row, Mapping) or not row:
        return None
    report = row.get("gp_training_report")
    if not isinstance(report, Mapping) or not report:
        return None
    time_steps = report.get("time_steps")
    if not isinstance(time_steps, (list, tuple)) or not time_steps:
        return None
    epochs = []
    early_stopped = []
    lr_reductions = []
    try:
        for step in time_steps:
            if not isinstance(step, Mapping):
                return None
            stopped_early = step["stopped_early"]
            if not isinstance(stopped_early, (bool, np.bool_)):
                return None
            epochs.append(int(step["epochs_trained"]))
            early_stopped.append(bool(stopped_early))
            lr_reductions.append(int(step["lr_reductions"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return {
        "num_models": len(time_steps),
        "num_early_stopped": sum(early_stopped),
        "epochs_mean": float(np.mean(epochs)),
        "epochs_min": min(epochs),
        "epochs_max": max(epochs),
        "lr_reductions": sum(lr_reductions),
    }


@torch.no_grad()
def rollout_pause_bundle(
    *,
    model,
    bundle,
    policy,
    temperature,
    log_prior,
    is_llr_as_sufficient_statistic,
    device,
    num_classes,
):
    device = torch.device(device)
    windows, epochs, patches = bundle.x.shape[:3]
    log_prior = torch.as_tensor(
        log_prior,
        dtype=torch.float32,
        device=device,
    )
    if log_prior.shape != (num_classes,) or not torch.isfinite(log_prior).all():
        raise ValueError("target log_prior must be finite [K]")
    if temperature is not None:
        temperature = torch.as_tensor(
            temperature,
            dtype=torch.float32,
            device=device,
        )
        if temperature.shape != (patches,):
            raise ValueError(f"temperature must be [{patches}]")
        if not torch.isfinite(temperature).all() or torch.any(temperature <= 0):
            raise ValueError("temperature must be finite and positive")

    horizon = patches if policy.upper_horizon is None else policy.upper_horizon
    if not 1 <= horizon <= patches:
        raise ValueError(f"policy horizon must be in [1, {patches}]")

    model.eval()
    all_labels = []
    all_predictions = []
    all_hitting_time = []
    all_posteriors = []
    end_idx_traces = []

    for window_index in range(windows):
        x = torch.as_tensor(
            bundle.x[window_index : window_index + 1],
            dtype=torch.float32,
            device=device,
        )
        end_idx = torch.full(
            (1, epochs),
            patches - 1,
            dtype=torch.long,
            device=device,
        )
        for epoch_position in range(epochs):
            logits = model(x, end_idx)
            if logits.shape != (1, epochs, patches, num_classes):
                raise ValueError(
                    "PAUSE logits must be "
                    f"[1,{epochs},{patches},{num_classes}], got "
                    f"{tuple(logits.shape)}"
                )
            epoch_logits = logits[:, epoch_position]
            epoch_label = torch.as_tensor(
                [bundle.y[window_index, epoch_position]],
                dtype=torch.long,
                device=device,
            )
            llrs, _, _, _, _, posteriors = extract_llrs_from_logits(
                epoch_logits,
                epoch_label,
                log_prior,
                temperature,
                return_posteriors=True,
            )
            hitting_time, prediction = decision_helper(
                policy.model_pool,
                policy.likelihood_pool,
                policy.cost_matrix,
                policy.cost,
                llrs,
                is_llr_as_sufficient_statistic=(
                    is_llr_as_sufficient_statistic
                ),
                posteriors=posteriors,
                upper_horizon=policy.upper_horizon,
            )
            hit_value = float(hitting_time.reshape(-1)[0].item())
            if not hit_value.is_integer():
                raise ValueError(
                    f"GP hitting time must be an integer, got {hit_value}"
                )
            hit = int(hit_value)
            if not 1 <= hit <= horizon:
                raise ValueError(
                    f"GP hitting time {hit} is outside policy horizon [1, {horizon}]"
                )
            prediction = prediction.reshape(-1)[0].long()
            if not 0 <= int(prediction.item()) < num_classes:
                raise ValueError("GP prediction is outside [0, K)")

            end_idx[0, epoch_position] = hit - 1
            all_labels.append(epoch_label.cpu())
            all_predictions.append(prediction.reshape(1).cpu())
            all_hitting_time.append(torch.tensor([hit], dtype=torch.long))
            all_posteriors.append(posteriors.cpu())
        end_idx_traces.append(end_idx.cpu())

    labels = torch.cat(all_labels, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    hitting_time = torch.cat(all_hitting_time, dim=0)
    posteriors = torch.cat(all_posteriors, dim=0)
    end_idx_trace = torch.cat(end_idx_traces, dim=0)
    occurrence_index = torch.as_tensor(
        bundle.occurrence_index.reshape(-1),
        dtype=torch.long,
    )
    keep = occurrence_index.eq(0)
    window_id = np.broadcast_to(
        bundle.window_id[:, None],
        (windows, epochs),
    ).reshape(-1)

    result = {
        "labels": labels[keep],
        "predictions": predictions[keep],
        "hitting_time": hitting_time[keep],
        "posteriors": posteriors[keep],
        "subject_id": torch.as_tensor(bundle.subject_id.reshape(-1))[keep],
        "window_id": torch.as_tensor(window_id.copy())[keep],
        "source_epoch_index": torch.as_tensor(
            bundle.source_epoch_index.reshape(-1)
        )[keep],
        "occurrence_index": occurrence_index[keep],
        "epoch_position": torch.as_tensor(bundle.epoch_position.reshape(-1))[keep],
        "all_labels": labels,
        "all_predictions": predictions,
        "all_hitting_time": hitting_time,
        "all_posteriors": posteriors,
        "keep_mask": keep,
        "end_idx_trace": end_idx_trace,
        "history_mode": "closed_loop_block32",
    }
    return result


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _checkpoint_identity(payload, dataset):
    """Resolve checkpoint identity to GP's zero-based canonical fold id.

    DREAMS PAUSE checkpoints historically stored ``fold_id`` as the one-based
    test-subject number, while GP artifacts use zero-based fold ids.  Newer
    payloads may instead store an already canonical ``fold_id``.  When a
    ``test_subject`` is present it disambiguates both conventions.
    """
    candidates = []
    if dataset == "dreams" and "test_subject" in payload:
        test_subject = int(payload["test_subject"])
        canonical_subject = test_subject - 1
        candidates.append(canonical_subject)
        if "fold_id" in payload:
            fold_id = int(payload["fold_id"])
            candidates.append(
                fold_id - 1 if fold_id == test_subject else fold_id
            )
    elif "fold_id" in payload:
        candidates.append(int(payload["fold_id"]))
    if not candidates:
        required = "test_subject or fold_id" if dataset == "dreams" else "fold_id"
        raise ValueError(f"checkpoint missing {required} identity")
    if len(set(candidates)) != 1:
        raise ValueError(
            f"checkpoint has conflicting identity metadata: {candidates}"
        )
    return candidates[0]


class PauseGPBackend:
    name = "pause"

    def _load_target_context(self, *, profile, fold, pipeline):
        identity = identity_for_fold(profile.dataset, fold)
        payload, checkpoint_path = load_checkpoint_payload(
            profile,
            fold,
            device=pipeline.config.device,
            strict=True,
        )
        if not isinstance(payload, dict):
            raise ValueError("PAUSE checkpoint payload must be a mapping")
        actual_variant = str(payload.get("model_variant", "paired"))
        if actual_variant != profile.model_variant:
            raise ValueError(
                f"checkpoint model_variant {actual_variant!r} != requested "
                f"{profile.model_variant!r}"
            )
        stored_identity = _checkpoint_identity(payload, profile.dataset)
        if stored_identity != identity.canonical_fold_id:
            raise ValueError(
                f"checkpoint identity {stored_identity} != target "
                f"{identity.canonical_fold_id}"
            )

        try:
            model_kwargs = dict(payload["model_kwargs"])
            contract = (
                int(model_kwargs["num_epochs"]),
                int(model_kwargs["num_patches"]),
                int(model_kwargs["num_classes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint model contract is incomplete") from exc
        if contract != (32, 30, 5):
            raise ValueError(
                f"checkpoint model contract {contract} != expected (32, 30, 5)"
            )

        pause_fold = assemble_fold(fold, num_epochs=32, num_classes=5)
        try:
            checkpoint_stats = np.asarray(
                [payload["train_mean"], payload["train_std"]],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise ValueError("checkpoint missing normalization statistics") from exc
        fold_stats = np.asarray(
            [pause_fold.train_mean, pause_fold.train_std],
            dtype=np.float64,
        )
        if (
            not np.isfinite(checkpoint_stats).all()
            or not np.isfinite(fold_stats).all()
            or not np.allclose(
                checkpoint_stats,
                fold_stats,
                atol=1e-4,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "checkpoint normalization statistics do not match target fold"
            )

        if "model_state" not in payload:
            raise ValueError("checkpoint missing model_state")
        model = PAUSE(**model_kwargs).to(pipeline.config.device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()

        class_counts = np.asarray(pause_fold.class_counts, dtype=np.float64)
        if class_counts.shape != (5,) or not np.all(class_counts > 0):
            raise ValueError("target fold class_counts must be positive [5]")
        target_prior = torch.as_tensor(
            class_counts / class_counts.sum(),
            dtype=torch.float32,
        )
        target_log_prior = target_prior.log()
        return (
            model,
            pause_fold.test,
            target_log_prior,
            str(checkpoint_path),
            sha256_file(checkpoint_path),
        )

    def evaluate_target_policy(
        self,
        *,
        policy,
        model,
        bundle,
        temperature,
        log_prior,
        pipeline,
    ):
        rollout = rollout_pause_bundle(
            model=model,
            bundle=bundle,
            policy=policy,
            temperature=temperature,
            log_prior=log_prior,
            is_llr_as_sufficient_statistic=(
                pipeline.config.is_llr_as_sufficient_statistic
            ),
            device=pipeline.config.device,
            num_classes=pipeline.config.num_classes,
        )
        metrics = pipeline.summarize_policy_outputs(
            labels=rollout["labels"],
            predictions=rollout["predictions"],
            hitting_time=rollout["hitting_time"],
            posteriors=rollout["posteriors"],
            cost_matrix=policy.cost_matrix,
        )

        for key in (
            "labels",
            "end_idx_trace",
            "subject_id",
            "window_id",
            "source_epoch_index",
            "occurrence_index",
            "epoch_position",
            "all_hitting_time",
        ):
            if key not in rollout:
                continue
            value = rollout[key]
            metrics[key] = value.cpu().numpy() if torch.is_tensor(value) else value
        metrics["history_mode"] = rollout["history_mode"]
        return metrics

    def _write_manifests(
        self,
        *,
        profile,
        identity,
        source_fold_ids,
        training_shards,
        history_draws,
        checkpoint_path,
        checkpoint_sha256,
        temperature_artifact,
        pipeline,
        output_run_dir,
        results,
        selected_fold_ids,
    ):
        output_run_dir = Path(output_run_dir)
        history_seeds = list(profile.history_seeds[:history_draws])
        use_temperature = temperature_artifact.temperature is not None
        source_artifacts = {}
        for shard in training_shards:
            source_id = int(shard.canonical_fold_id)
            values = {
                "oof_path": shard.oof_path,
                "oof_sha256": shard.oof_sha256,
                "checkpoint_path": shard.checkpoint_path,
                "checkpoint_sha256": shard.checkpoint_sha256,
            }
            if any(value is None for value in values.values()):
                raise ValueError(
                    f"source fold {source_id} is missing artifact provenance"
                )
            source_artifacts[str(source_id)] = values
        if set(map(int, source_artifacts)) != set(source_fold_ids):
            raise ValueError("source artifact provenance does not match source folds")
        per_cost_timing = {
            str(cost): {
                key: float(row[key])
                for key in (
                    "training_seconds",
                    "evaluation_seconds",
                    "total_seconds",
                )
            }
            for cost, row in results.items()
            if row is not None
            and all(
                key in row
                for key in (
                    "training_seconds",
                    "evaluation_seconds",
                    "total_seconds",
                )
            )
        }
        per_cost_early_stopping = {
            str(cost): summary
            for cost, row in results.items()
            if row is not None
            and (summary := _early_stopping_summary(row)) is not None
        }
        cost_metadata_values = []
        for row in results.values():
            if row is None or row.get("cost_metadata") is None:
                continue
            metadata = row["cost_metadata"]
            if not isinstance(metadata, Mapping):
                raise ValueError("cost_metadata must be a mapping")
            cost_metadata_values.append(dict(metadata))
        misclassification_cost = None
        if cost_metadata_values:
            misclassification_cost = cost_metadata_values[0]
            if any(
                metadata != misclassification_cost
                for metadata in cost_metadata_values[1:]
            ):
                raise ValueError("inconsistent cost metadata across policies")
        early_stopping_config = {
            "gp_early_stopping": bool(pipeline.config.gp_early_stopping),
            "gp_validation_fraction": float(
                pipeline.config.gp_validation_fraction
            ),
            "gp_min_epochs": int(pipeline.config.gp_min_epochs),
            "gp_patience": int(pipeline.config.gp_patience),
            "gp_min_delta": float(pipeline.config.gp_min_delta),
            "gp_learning_rate": float(pipeline.config.gp_learning_rate),
            "gp_lr_reduce_factor": float(pipeline.config.gp_lr_reduce_factor),
            "gp_max_lr_reductions": int(
                pipeline.config.gp_max_lr_reductions
            ),
            "gp_early_stopping_seed": int(
                pipeline.config.gp_early_stopping_seed
            ),
        }
        statistic = (
            "llr"
            if pipeline.config.is_llr_as_sufficient_statistic
            else "posterior"
        )
        fold_manifest = {
            "dataset": profile.dataset,
            "eval_backend": self.name,
            "target_fold": identity.canonical_fold_id,
            "source_folds": list(source_fold_ids),
            "source_artifacts": source_artifacts,
            "history_draws": history_draws,
            "history_seeds": history_seeds,
            "history_mode": "closed_loop_block32",
            "model_variant": profile.model_variant,
            "calibration_history_mode": profile.calibration_history_mode,
            "num_history_draws": history_draws,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "temperature_path": temperature_artifact.path,
            "temperature_sha256": temperature_artifact.sha256,
            "use_temperature": use_temperature,
            "upper_horizon": pipeline.config.upper_horizon,
            "cost_pool": list(pipeline.config.cost_pool),
            "penalty": float(pipeline.config.penalty),
            "penalty_alpha": float(pipeline.config.penalty_alpha),
            "misclassification_cost": misclassification_cost,
            "gp_batch_size": pipeline.config.gp_batch_size,
            "gp_epochs": pipeline.config.gp_epochs,
            "num_inducing_points": pipeline.config.num_inducing_points,
            "posterior_device": getattr(pipeline.config, "posterior_device", "cpu"),
            "gp_statistic": statistic,
            "selected_folds": list(selected_fold_ids),
            "per_cost_timing": per_cost_timing,
            "per_cost_early_stopping": per_cost_early_stopping,
            **early_stopping_config,
        }
        fold_manifest_path = output_run_dir / f"{identity.fold_stem}_manifest.json"
        _atomic_write_json(fold_manifest_path, fold_manifest)

        run_manifest_path = output_run_dir / "run_manifest.json"
        if run_manifest_path.is_file():
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        else:
            run_manifest = {}
        completed = set(run_manifest.get("completed_folds", []))
        completed.add(identity.canonical_fold_id)
        fold_artifacts = dict(run_manifest.get("fold_artifacts", {}))
        fold_artifacts[str(identity.canonical_fold_id)] = fold_manifest
        fold_timings = dict(run_manifest.get("fold_timings", {}))
        fold_timings[str(identity.canonical_fold_id)] = per_cost_timing
        fold_early_stopping = dict(run_manifest.get("fold_early_stopping", {}))
        fold_early_stopping[str(identity.canonical_fold_id)] = (
            per_cost_early_stopping
        )
        run_manifest.update(
            {
                "dataset": profile.dataset,
                "eval_backend": self.name,
                "cost_pool": list(pipeline.config.cost_pool),
                "penalty": pipeline.config.penalty,
                "penalty_alpha": pipeline.config.penalty_alpha,
                "gp_epochs": pipeline.config.gp_epochs,
                "gp_batch_size": pipeline.config.gp_batch_size,
                "num_inducing_points": pipeline.config.num_inducing_points,
                "gp_statistic": statistic,
                "model_variant": profile.model_variant,
                "calibration_history_mode": profile.calibration_history_mode,
                "num_history_draws": history_draws,
                "history_seeds": history_seeds,
                "selected_folds": list(selected_fold_ids),
                "device": str(pipeline.config.device),
                "posterior_device": getattr(
                    pipeline.config,
                    "posterior_device",
                    "cpu",
                ),
                "upper_horizon": pipeline.config.upper_horizon,
                "completed_folds": sorted(completed),

                "fold_artifacts": fold_artifacts,
                "fold_timings": fold_timings,
                "fold_early_stopping": fold_early_stopping,
                **early_stopping_config,
            }
        )
        _atomic_write_json(run_manifest_path, run_manifest)

    def run_fold(
        self,
        *,
        profile,
        fold,
        all_folds,
        pipeline,
        logger,
        smoke,
        checkpoint_run_dir,
        output_run_dir,
        selected_fold_ids=None,
    ):
        history_draws = profile.num_history_draws
        all_folds = tuple(all_folds)
        identity = identity_for_fold(profile.dataset, fold)
        selected_fold_ids = (
            (identity.canonical_fold_id,)
            if selected_fold_ids is None
            else tuple(int(fold_id) for fold_id in selected_fold_ids)
        )
        if identity.canonical_fold_id not in selected_fold_ids:
            raise ValueError("selected_fold_ids must contain the target fold")
        (
            model,
            test_bundle,
            target_log_prior,
            checkpoint_path,
            checkpoint_sha256,
        ) = self._load_target_context(
            profile=profile,
            fold=fold,
            pipeline=pipeline,
        )
        training_shards = load_pause_training_shards(
            profile,
            target_fold=fold,
            all_folds=all_folds,
            history_draws=history_draws,
        )
        temperature_artifact = load_pause_target_temperature(
            profile,
            fold=fold,
            all_folds=all_folds,
        )
        source_fold_ids = []
        for source in all_folds:
            source_id = identity_for_fold(
                profile.dataset,
                source,
            ).canonical_fold_id
            if source_id != identity.canonical_fold_id:
                source_fold_ids.append(source_id)
        source_fold_ids = tuple(sorted(source_fold_ids))

        def evaluator(policy):
            return self.evaluate_target_policy(
                policy=policy,
                model=model,
                bundle=test_bundle,
                temperature=temperature_artifact.temperature,
                log_prior=target_log_prior,
                pipeline=pipeline,
            )

        results = pipeline.run_policy_sweep(
            training_shards=training_shards,
            temperature=temperature_artifact.temperature,
            evaluator=evaluator,
            fold_name=identity.fold_stem,
            logger=logger,
            smoke=smoke,
            checkpoint_run_dir=checkpoint_run_dir,
            output_run_dir=output_run_dir,
        )
        self._write_manifests(
            profile=profile,
            identity=identity,
            source_fold_ids=source_fold_ids,
            training_shards=training_shards,
            history_draws=history_draws,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            temperature_artifact=temperature_artifact,
            pipeline=pipeline,
            output_run_dir=output_run_dir,
            results=results,
            selected_fold_ids=selected_fold_ids,
        )
        logger.info(
            "[S2] %s PAUSE-GP sources=%d history_draws=%d temperature=%s",
            identity.fold_stem,
            len(source_fold_ids),
            history_draws,
            "on" if temperature_artifact.temperature is not None else "off",
        )
        return results
