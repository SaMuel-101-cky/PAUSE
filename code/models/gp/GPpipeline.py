import os
import time
from copy import deepcopy
from numbers import Integral

import numpy as np
import torch
from sklearn.metrics import f1_score

from .backward_induction import GP_training_routine, G_st
from config.gp import GPConfig
from .llr_utils import calibrated_posteriors_from_logits, extract_llrs_from_logits
from .metrics import calc_sns
from .contracts import GPPolicy, GPTrainingStates
from .costs import build_class_weighted_cost_matrix, validate_penalty_alpha
from .class_metrics import summarize_per_class_outputs


def _validate_integer_config(name, value, *, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")


def average_a_posterior_risk(
    posteriors: torch.Tensor,
    hitting_time: torch.Tensor,
    penalty: float,
) -> float:
    stopping_risk, _ = G_st(posteriors, penalty)
    hitting_time = torch.as_tensor(hitting_time, device=posteriors.device).long().view(-1)
    if posteriors.ndim != 3:
        raise ValueError(f"expected posteriors [N,T,K], got {posteriors.shape}")
    if len(hitting_time) != posteriors.shape[0]:
        raise ValueError(
            f"hitting_time length {len(hitting_time)} != posteriors batch {posteriors.shape[0]}"
        )
    time_index = torch.clamp(hitting_time - 1, min=0, max=posteriors.shape[1] - 1)
    sample_index = torch.arange(posteriors.shape[0], device=posteriors.device)
    return float(stopping_risk[sample_index, time_index].mean().item())


class GPPipeline:
    """Fold-level orchestration for gp early-stop evaluation."""

    def __init__(self, config: GPConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if not self.config.cost_pool:
            raise ValueError("cost_pool must not be empty")
        posterior_device = getattr(self.config, "posterior_device", "cpu")
        if posterior_device not in ("cpu", "gpu"):
            raise ValueError("posterior_device must be 'cpu' or 'gpu'")
        if self.config.upper_horizon is not None and self.config.upper_horizon < 2:
            raise ValueError("upper_horizon must be at least 2")
        if not np.isfinite(self.config.penalty) or self.config.penalty < 0:
            raise ValueError("penalty must be finite and non-negative")
        self.config.penalty_alpha = validate_penalty_alpha(
            self.config.penalty_alpha
        )
        if not 0 < self.config.gp_validation_fraction < 1:
            raise ValueError("gp_validation_fraction must be in (0, 1)")
        _validate_integer_config(
            "gp_min_epochs", self.config.gp_min_epochs, minimum=1
        )
        _validate_integer_config("gp_patience", self.config.gp_patience, minimum=1)
        if not np.isfinite(self.config.gp_min_delta) or self.config.gp_min_delta < 0:
            raise ValueError("gp_min_delta must be finite and non-negative")
        if (
            not np.isfinite(self.config.gp_learning_rate)
            or self.config.gp_learning_rate <= 0
        ):
            raise ValueError("gp_learning_rate must be finite and positive")
        if not 0 < self.config.gp_lr_reduce_factor < 1:
            raise ValueError("gp_lr_reduce_factor must be in (0, 1)")
        _validate_integer_config(
            "gp_max_lr_reductions",
            self.config.gp_max_lr_reductions,
            minimum=0,
        )
        _validate_integer_config(
            "gp_early_stopping_seed",
            self.config.gp_early_stopping_seed,
            minimum=0,
        )

    def _posterior_storage_device(self) -> torch.device:
        posterior_device = getattr(self.config, "posterior_device", "cpu")
        if posterior_device == "cpu":
            return torch.device("cpu")
        training_device = torch.device(self.config.device)
        if training_device.type != "cuda":
            raise ValueError(
                "posterior_device='gpu' requires GPConfig.device to be a CUDA device"
            )
        return training_device

    def prepare_training_states(self, shards, temperature):
        shards = tuple(shards)
        if not shards:
            raise ValueError("training shards must not be empty")
        storage_device = self._posterior_storage_device()
        llr_parts = []
        posterior_parts = []
        label_parts = []
        for shard in shards:
            logits = shard.logits.to(storage_device)
            log_prior = shard.log_prior.to(storage_device)
            if self.config.is_llr_as_sufficient_statistic:
                llrs, labels, _, _, _, posteriors = extract_llrs_from_logits(
                    logits,
                    shard.labels,
                    log_prior,
                    temperature,
                    return_posteriors=True,
                )
                llr_parts.append(llrs)
            else:
                labels = shard.labels
                posteriors = calibrated_posteriors_from_logits(
                    logits,
                    log_prior,
                    temperature,
                )
            posterior_parts.append(posteriors)
            label_parts.append(labels)
        return GPTrainingStates(
            llrs=(torch.cat(llr_parts, dim=0) if llr_parts else None),
            posteriors=torch.cat(posterior_parts, dim=0),
            labels=torch.cat(label_parts, dim=0),
            device=storage_device,
        )

    def fit_policy(
        self,
        *,
        training_states,
        cost,
        fold_name,
        smoke,
        checkpoint_run_dir,
        output_run_dir,
    ):
        if checkpoint_run_dir is None or output_run_dir is None:
            raise ValueError("checkpoint_run_dir and output_run_dir are required")
        cost_matrix, cost_metadata = build_class_weighted_cost_matrix(
            training_states.labels,
            num_classes=self.config.num_classes,
            penalty=self.config.penalty,
            alpha=self.config.penalty_alpha,
        )
        if self.config.is_llr_as_sufficient_statistic and training_states.llrs is None:
            raise ValueError("LLR statistic mode requires training-state llrs")
        num_training_rows = training_states.labels.shape[0]
        llrs = (
            training_states.llrs
            if self.config.is_llr_as_sufficient_statistic
            else None
        )
        model_pool, likelihood_pool, training_report = GP_training_routine(
            llrs,
            cost_matrix,
            cost,
            subproject=fold_name,
            num_epochs=2 if smoke else self.config.gp_epochs,
            batch_size=min(
                self.config.gp_batch_size,
                num_training_rows,
            ),
            num_inducing_points=min(
                self.config.num_inducing_points,
                num_training_rows,
            ),
            is_llr_as_sufficient_statistic=(
                self.config.is_llr_as_sufficient_statistic
            ),
            savedir=checkpoint_run_dir,
            output_savedir=output_run_dir,
            posteriors=training_states.posteriors,
            device=self.config.device,
            upper_horizon=self.config.upper_horizon,
            gp_early_stopping=self.config.gp_early_stopping,
            gp_validation_fraction=self.config.gp_validation_fraction,
            gp_min_epochs=self.config.gp_min_epochs,
            gp_patience=self.config.gp_patience,
            gp_min_delta=self.config.gp_min_delta,
            gp_learning_rate=self.config.gp_learning_rate,
            gp_lr_reduce_factor=self.config.gp_lr_reduce_factor,
            gp_max_lr_reductions=self.config.gp_max_lr_reductions,
            gp_early_stopping_seed=self.config.gp_early_stopping_seed,
            posterior_device=getattr(self.config, "posterior_device", "cpu"),
            return_training_report=True,
        )
        return GPPolicy(
            model_pool=tuple(model_pool),
            likelihood_pool=tuple(likelihood_pool),
            cost=cost,
            upper_horizon=self.config.upper_horizon,
            cost_matrix=cost_matrix,
            cost_metadata=cost_metadata,
            training_report=training_report,
        )

    def run_policy_sweep(
        self,
        *,
        training_shards,
        temperature,
        evaluator,
        fold_name,
        logger,
        smoke,
        checkpoint_run_dir,
        output_run_dir,
    ):
        training_states = self.prepare_training_states(
            training_shards,
            temperature,
        )
        results = {}
        for cost in self.config.cost_pool:
            try:
                training_started = time.perf_counter()
                policy = self.fit_policy(
                    training_states=training_states,
                    cost=cost,
                    fold_name=fold_name,
                    smoke=smoke,
                    checkpoint_run_dir=checkpoint_run_dir,
                    output_run_dir=output_run_dir,
                )
                training_seconds = time.perf_counter() - training_started
                evaluation_started = time.perf_counter()
                row = dict(evaluator(policy))
                evaluation_seconds = time.perf_counter() - evaluation_started
                row["cost_metadata"] = deepcopy(policy.cost_metadata)
                row.update(
                    training_seconds=training_seconds,
                    evaluation_seconds=evaluation_seconds,
                    total_seconds=training_seconds + evaluation_seconds,
                )
                if policy.training_report is not None:
                    row["gp_training_report"] = policy.training_report
                results[cost] = row
            except Exception:
                logger.exception("[S2] %s cost=%s failed", fold_name, cost)
                results[cost] = None
        return results

    def summarize_policy_outputs(
        self,
        *,
        labels,
        predictions,
        hitting_time,
        posteriors,
        cost_matrix=None,
    ):
        labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
        predictions = torch.as_tensor(
            predictions,
            dtype=torch.long,
            device="cpu",
        )
        hitting_time = torch.as_tensor(
            hitting_time,
            dtype=torch.long,
            device="cpu",
        )
        posteriors = torch.as_tensor(posteriors, device="cpu")
        if predictions.shape != labels.shape or hitting_time.shape != labels.shape:
            raise ValueError("labels, predictions, and hitting_time must align")
        if posteriors.ndim != 3 or posteriors.shape[0] != labels.shape[0]:
            raise ValueError("posteriors must be [N,T,K] aligned with labels")
        macro_sensitivity = float(
            calc_sns(labels, predictions, self.config.num_classes)
        )
        labels_np = labels.numpy()
        predictions_np = predictions.numpy()
        return {
            "mean_hitting_time": float(hitting_time.float().mean().item()),
            "accuracy": float(labels.eq(predictions).float().mean().item()),
            "macro_sensitivity": macro_sensitivity,
            "macro_f1": float(
                f1_score(
                    labels_np,
                    predictions_np,
                    labels=list(range(self.config.num_classes)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "aapr": average_a_posterior_risk(
                posteriors,
                hitting_time,
                self.config.penalty if cost_matrix is None else cost_matrix,
            ),
            "mean_per_class_error_rate": 1.0 - macro_sensitivity,
            "labels": labels_np,
            "predictions": predictions_np,
            "hitting_time": hitting_time.numpy(),
            "per_class": summarize_per_class_outputs(
                labels_np,
                predictions_np,
                hitting_time.numpy(),
                self.config.num_classes,
            ),
        }

    def print_summary(
        self,
        all_gp_results,
        logger,
        *,
        output_run_dir,
        expected_folds=None,
    ):
        logger.info("=" * 70)
        logger.info("[S2] gp early-stopping summary (per-fold average)")
        logger.info("=" * 70)
        for cost in self.config.cost_pool:
            accs, mhts, mf1s, snss = [], [], [], []
            labels, predictions, hitting_times = [], [], []
            for _subject, results in all_gp_results.items():
                row = results.get(cost) if results else None
                if row is None:
                    continue
                accs.append(row["accuracy"])
                mhts.append(row["mean_hitting_time"])
                mf1s.append(row["macro_f1"])
                snss.append(row["macro_sensitivity"])
                if all(
                    key in row for key in ("labels", "predictions", "hitting_time")
                ):
                    labels.append(np.asarray(row["labels"]))
                    predictions.append(np.asarray(row["predictions"]))
                    hitting_times.append(np.asarray(row["hitting_time"]))
            if accs:
                logger.info(
                    f"  cost={cost:5.6f}  n_folds={len(accs):2d}  "
                    f"acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}  "
                    f"MF1={np.mean(mf1s):.4f}+/-{np.std(mf1s):.4f}  "
                    f"macro_sns={np.mean(snss):.4f}+/-{np.std(snss):.4f}  "
                    f"mean_HT={np.mean(mhts):5.1f}+/-{np.std(mhts):5.1f}"
                )
                if labels:
                    per_class = summarize_per_class_outputs(
                        np.concatenate(labels),
                        np.concatenate(predictions),
                        np.concatenate(hitting_times),
                        self.config.num_classes,
                    )
                    for class_id, metrics in per_class.items():
                        mean_hitting_time = metrics["mean_hitting_time"]
                        mean_hitting_time_text = (
                            f"{mean_hitting_time:.2f}"
                            if mean_hitting_time is not None
                            else "N/A"
                        )
                        logger.info(
                            "  cost=%.6f  class=%s  support=%d  mean_HT=%s  "
                            "recall=%.4f  precision=%.4f  mean_error_rate=%.4f",
                            cost,
                            class_id,
                            metrics["support"],
                            mean_hitting_time_text,
                            metrics["recall"],
                            metrics["precision"],
                            metrics["mean_error_rate"],
                        )

        os.makedirs(output_run_dir, exist_ok=True)
        arrays = {
            "all_results": all_gp_results,
            "cost_pool": self.config.cost_pool,
            "completed_folds": np.asarray(len(all_gp_results), dtype=np.int64),
        }
        if expected_folds is not None:
            arrays["expected_folds"] = np.asarray(expected_folds, dtype=np.int64)
        np.savez(os.path.join(output_run_dir, "gp_results_summary.npz"), **arrays)
