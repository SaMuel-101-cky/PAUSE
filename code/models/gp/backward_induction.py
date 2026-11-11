import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gpytorch
import torch
from loguru import logger
from termcolor import colored
from tqdm import tqdm

from .early_stopping import EarlyStoppingController, make_validation_split
from .costs import expected_misclassification_costs
from .GPmodel import GPModel
from .io_utils import save_gp_model
from .llr_utils import (
    _truncate_to_upper_horizon,
    get_upper_triangle,
    retrieve_posteriors_from_llrm,
)


@dataclass(frozen=True)
class GPFitMetadata:
    epochs_trained: int
    best_epoch: int
    best_validation_loss: float | None
    stopped_early: bool
    lr_reductions: int
    final_learning_rate: float
    train_size: int
    validation_size: int
    inducing_points: int


@dataclass(frozen=True)
class GPFitResult:
    model: Any
    likelihood: Any
    batch_loss: tuple[float, ...]
    epoch_train_loss: tuple[float, ...]
    epoch_validation_loss: tuple[float, ...]
    metadata: GPFitMetadata


def GP_training_routine(
    llrs,
    penalty,
    cost,
    subproject,
    *,
    num_epochs,
    batch_size,
    num_inducing_points,
    is_llr_as_sufficient_statistic,
    savedir,
    output_savedir,
    posteriors,
    device=None,
    upper_horizon,
    gp_early_stopping,
    gp_validation_fraction,
    gp_min_epochs,
    gp_patience,
    gp_min_delta,
    gp_learning_rate,
    gp_lr_reduce_factor,
    gp_max_lr_reductions,
    gp_early_stopping_seed,
    posterior_device="cpu",
    return_training_report=False,
):
    llrs, posteriors, horizon = _truncate_to_upper_horizon(
        llrs, posteriors, upper_horizon
    )
    if llrs is not None:
        shapes = llrs.shape
        assert shapes[2] == shapes[3]
    elif is_llr_as_sufficient_statistic:
        raise ValueError("LLR statistic mode requires llrs")
    elif posteriors is None:
        raise ValueError("posterior statistic mode requires posteriors")

    horizon_suffix = "" if upper_horizon is None else f"_horizon{horizon}"
    penalty_tensor = torch.as_tensor(penalty)
    penalty_descriptor = (
        "costmatrix"
        if penalty_tensor.ndim == 2
        else f"penalty{float(penalty_tensor.item())}"
    )
    descriptor = (
        f"{penalty_descriptor}_cost{cost}_batchsize{batch_size}"
        f"_indpnt{num_inducing_points}_epoch{num_epochs}_isllrs{is_llr_as_sufficient_statistic}"
        f"{horizon_suffix}"
    )
    model_dirname = Path(savedir) / subproject / descriptor
    artifact_root = Path(output_savedir) if output_savedir is not None else Path(savedir)
    artifact_dirname = artifact_root / subproject / descriptor
    model_dirname.mkdir(parents=True, exist_ok=True)
    artifact_dirname.mkdir(parents=True, exist_ok=True)

    result = backward_induction(
        llrs,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_inducing_points=num_inducing_points,
        penalty=penalty,
        cost=cost,
        is_llr_as_sufficient_statistic=is_llr_as_sufficient_statistic,
        device=device,
        posteriors=posteriors,
        upper_horizon=upper_horizon,
        gp_early_stopping=gp_early_stopping,
        gp_validation_fraction=gp_validation_fraction,
        gp_min_epochs=gp_min_epochs,
        gp_patience=gp_patience,
        gp_min_delta=gp_min_delta,
        gp_learning_rate=gp_learning_rate,
        gp_lr_reduce_factor=gp_lr_reduce_factor,
        gp_max_lr_reductions=gp_max_lr_reductions,
        gp_early_stopping_seed=gp_early_stopping_seed,
        posterior_device=posterior_device,
    )
    if len(result) == 6:
        (
            continuation_risk,
            stopping_risk,
            predicted_labels,
            loss_pool,
            model_pool,
            likelihood_pool,
        ) = result
        epoch_train_loss_pool = loss_pool
        epoch_validation_loss_pool = [()] * len(loss_pool)
        training_report = {
            "early_stopping_enabled": False,
            "time_steps": [],
        }
    else:
        (
            continuation_risk,
            stopping_risk,
            predicted_labels,
            loss_pool,
            epoch_train_loss_pool,
            epoch_validation_loss_pool,
            model_pool,
            likelihood_pool,
            training_report,
        ) = result
    save_gp_model(
        model_pool,
        likelihood_pool,
        continuation_risk,
        stopping_risk,
        predicted_labels,
        loss_pool,
        model_dirname,
        artifact_dirname,
        epoch_train_loss_pool=epoch_train_loss_pool,
        epoch_validation_loss_pool=epoch_validation_loss_pool,
        training_report=training_report,
        max_epochs=num_epochs,
    )

    if return_training_report:
        return model_pool, likelihood_pool, training_report
    return model_pool, likelihood_pool


@logger.catch(reraise=True)
def backward_induction(
    llrs,
    num_epochs,
    batch_size,
    num_inducing_points,
    *,
    penalty,
    cost,
    is_llr_as_sufficient_statistic,
    device=None,
    posteriors=None,
    upper_horizon,
    gp_early_stopping,
    gp_validation_fraction,
    gp_min_epochs,
    gp_patience,
    gp_min_delta,
    gp_learning_rate,
    gp_lr_reduce_factor,
    gp_max_lr_reductions,
    gp_early_stopping_seed,
    posterior_device="cpu",
):
    llrs, posteriors, _ = _truncate_to_upper_horizon(
        llrs, posteriors, upper_horizon
    )
    training_device = None if device is None else torch.device(device)
    if posterior_device not in ("cpu", "gpu"):
        raise ValueError("posterior_device must be 'cpu' or 'gpu'")
    if posterior_device == "gpu":
        if training_device is None or training_device.type != "cuda":
            raise ValueError(
                "posterior_device='gpu' requires a CUDA training device"
            )
        state_device = training_device
    else:
        state_device = torch.device("cpu")

    if llrs is None:
        if is_llr_as_sufficient_statistic:
            raise ValueError("LLR statistic mode requires llrs")
        if posteriors is None:
            raise ValueError("posterior statistic mode requires posteriors")
        data_num, time_steps, num_classes = posteriors.shape
        if training_device is None:
            training_device = posteriors.device
    else:
        data_num, time_steps, num_classes, _ = llrs.shape
        if training_device is None:
            training_device = llrs.device
        llrs = llrs.detach().to(state_device)

    if posteriors is None:
        posteriors = retrieve_posteriors_from_llrm(llrs)
    else:
        posteriors = posteriors.detach().to(state_device)
        assert posteriors.shape == (data_num, time_steps, num_classes)

    stopping_risk, predicted_labels = G_st(posteriors, penalty)
    continuation_risk = torch.zeros((data_num, time_steps), device=state_device)
    minimal_risk = torch.zeros((data_num, time_steps), device=state_device)
    continuation_risk[:, -1] = stopping_risk[:, -1]
    minimal_risk[:, -1] = stopping_risk[:, -1]

    validation_split = None
    if gp_early_stopping:
        validation_split = make_validation_split(
            data_num,
            gp_validation_fraction,
            gp_early_stopping_seed,
            state_device,
        )

    loss_pool = []
    epoch_train_loss_pool = []
    epoch_validation_loss_pool = []
    model_pool = []
    likelihood_pool = []
    time_step_reports = []
    for t in tqdm(
        reversed(range(time_steps - 1)),
        mininterval=5,
        desc=colored("Reverse time step progress: ", "blue"),
        colour="blue",
    ):
        G_min = minimal_risk[:, t + 1]
        if is_llr_as_sufficient_statistic:
            sufficient_statistic = get_upper_triangle(llrs)[:, t]
        else:
            sufficient_statistic = posteriors[:, t]

        if validation_split is None:
            training_statistic = sufficient_statistic
            training_target = G_min
            validation_statistic = None
            validation_target = None
        else:
            training_statistic = sufficient_statistic[
                validation_split.train_indices
            ]
            training_target = G_min[validation_split.train_indices]
            validation_statistic = sufficient_statistic[
                validation_split.validation_indices
            ]
            validation_target = G_min[validation_split.validation_indices]

        effective_inducing_points = min(
            num_inducing_points,
            len(training_statistic),
        )
        model, likelihood, _ = initialize_Gaussian_process_model(
            training_statistic,
            effective_inducing_points,
            device=training_device,
        )
        fit_result = stochastic_variational_ELBO_maximization(
            model,
            likelihood,
            training_statistic,
            training_target,
            num_epochs,
            batch_size,
            validation_stat=validation_statistic,
            validation_target=validation_target,
            early_stopping=gp_early_stopping,
            min_epochs=gp_min_epochs,
            patience=gp_patience,
            min_delta=gp_min_delta,
            learning_rate=gp_learning_rate,
            lr_reduce_factor=gp_lr_reduce_factor,
            max_lr_reductions=gp_max_lr_reductions,
        )
        model = fit_result.model
        likelihood = fit_result.likelihood
        loss_pool.append(fit_result.batch_loss)
        epoch_train_loss_pool.append(fit_result.epoch_train_loss)
        epoch_validation_loss_pool.append(fit_result.epoch_validation_loss)
        model_pool.append(model)
        likelihood_pool.append(likelihood)
        time_step_reports.append(
            {"reverse_time_step": t, **asdict(fit_result.metadata)}
        )

        cond_mean, _, _ = conditional_expectation_G_given_lmbda(
            model,
            likelihood,
            sufficient_statistic,
            batch_size=batch_size,
        )
        continuation_risk[:, t] = cond_mean.to(state_device) + cost
        minimal_risk[:, t] = torch.min(
            stopping_risk[:, t], continuation_risk[:, t]
        )

    return (
        continuation_risk,
        stopping_risk,
        predicted_labels,
        loss_pool,
        epoch_train_loss_pool,
        epoch_validation_loss_pool,
        model_pool,
        likelihood_pool,
        {
            "early_stopping_enabled": gp_early_stopping,
            "time_steps": time_step_reports,
        },
    )


def G_st(posteriors, penalty=1.0):
    """Compute the instantaneous stopping risk and corresponding label."""
    expected_cost = expected_misclassification_costs(posteriors, penalty)
    return torch.min(expected_cost, dim=-1)


def _module_device(module, fallback):
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device(fallback)


def initialize_Gaussian_process_model(stat, num_inducing_points=100, *, device=None):
    index_device = stat.device
    model_device = stat.device if device is None else torch.device(device)
    n_total = len(stat)
    permutation = torch.randperm(n_total)
    indices = permutation[:num_inducing_points].to(index_device)
    inducing_points = stat[indices].to(model_device)

    model = GPModel(inducing_points=inducing_points).to(model_device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(model_device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=0.02,
    )
    return model, likelihood, optimizer


def stochastic_variational_ELBO_maximization(
    model,
    likelihood,
    stat,
    G_min,
    num_epochs,
    batch_size,
    *,
    validation_stat=None,
    validation_target=None,
    early_stopping=False,
    min_epochs=5,
    patience=3,
    min_delta=1e-4,
    learning_rate=0.02,
    lr_reduce_factor=0.25,
    max_lr_reductions=1,
):
    assert len(stat) == len(G_min)
    if early_stopping:
        if validation_stat is None or validation_target is None:
            raise ValueError("validation data is required for early stopping")
        if len(validation_stat) != len(validation_target):
            raise ValueError("validation statistic and target sizes must match")
        if len(validation_stat) == 0:
            raise ValueError("validation data must not be empty")

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=learning_rate,
    )
    training_elbo = gpytorch.mlls.VariationalELBO(
        likelihood,
        model,
        num_data=G_min.size(0),
    )
    model_device = _module_device(model, stat.device)
    controller = None
    if early_stopping:
        controller = EarlyStoppingController(
            min(min_epochs, num_epochs),
            patience,
            min_delta,
            lr_reduce_factor,
            max_lr_reductions,
        )

    batch_loss = []
    epoch_train_loss = []
    epoch_validation_loss = []
    epochs_trained = 0
    stopped_early = False
    for epoch_index in range(num_epochs):
        model.train()
        likelihood.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        for start in range(0, len(stat), batch_size):
            x_batch = stat[start : start + batch_size].to(model_device)
            y_batch = G_min[start : start + batch_size].to(model_device)
            optimizer.zero_grad()
            loss = -training_elbo(model(x_batch), y_batch)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise ValueError("training loss must be finite")
            batch_loss.append(loss_value)
            batch_samples = len(y_batch)
            epoch_loss_sum += loss_value * batch_samples
            epoch_sample_count += batch_samples
            loss.backward()
            optimizer.step()

        epochs_trained = epoch_index + 1
        train_loss_mean = epoch_loss_sum / epoch_sample_count
        if not math.isfinite(train_loss_mean):
            raise ValueError("training loss must be finite")
        epoch_train_loss.append(train_loss_mean)

        action = "continue"
        validation_loss_mean = None
        if controller is not None:
            validation_loss_mean = _evaluate_validation_negative_elbo(
                model,
                likelihood,
                validation_stat,
                validation_target,
                batch_size,
            )
            if not math.isfinite(validation_loss_mean):
                raise ValueError("validation loss must be finite")
            epoch_validation_loss.append(validation_loss_mean)
            action = controller.observe(
                epochs_trained,
                validation_loss_mean,
                model,
                likelihood,
                optimizer,
                allow_lr_reduction=epochs_trained < num_epochs,
            )

        if (
            epochs_trained == 1
            or epochs_trained % 5 == 0
            or action in {"reduce_lr", "stop"}
        ):
            message = f"Epoch {epochs_trained}, train_loss={train_loss_mean:.6g}"
            if validation_loss_mean is not None:
                message += f", validation_loss={validation_loss_mean:.6g}"
            if action in {"reduce_lr", "stop"}:
                message += f", action={action}"
            logger.info(message)

        if action == "stop":
            stopped_early = epochs_trained < num_epochs
            break

    final_learning_rate = float(optimizer.param_groups[0]["lr"])
    if controller is not None:
        controller.restore_best(model, likelihood, optimizer)
        best_epoch = controller.best_epoch
        best_validation_loss = controller.best_validation_loss
        lr_reductions = controller.lr_reductions
    else:
        best_epoch = epochs_trained
        best_validation_loss = None
        lr_reductions = 0

    metadata = GPFitMetadata(
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
        lr_reductions=lr_reductions,
        final_learning_rate=final_learning_rate,
        train_size=len(stat),
        validation_size=len(validation_stat) if early_stopping else 0,
        inducing_points=int(model.variational_strategy.inducing_points.size(0)),
    )
    return GPFitResult(
        model=model,
        likelihood=likelihood,
        batch_loss=tuple(batch_loss),
        epoch_train_loss=tuple(epoch_train_loss),
        epoch_validation_loss=tuple(epoch_validation_loss),
        metadata=metadata,
    )


def _evaluate_validation_negative_elbo(
    model,
    likelihood,
    stat,
    target,
    batch_size,
):
    model.eval()
    likelihood.eval()
    validation_elbo = gpytorch.mlls.VariationalELBO(
        likelihood,
        model,
        num_data=target.size(0),
    )
    model_device = _module_device(model, stat.device)
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for start in range(0, len(stat), batch_size):
            x_batch = stat[start : start + batch_size].to(model_device)
            y_batch = target[start : start + batch_size].to(model_device)
            loss_value = float(
                (-validation_elbo(model(x_batch), y_batch)).item()
            )
            if not math.isfinite(loss_value):
                raise ValueError("validation loss must be finite")
            batch_samples = len(y_batch)
            loss_sum += loss_value * batch_samples
            sample_count += batch_samples
    validation_loss = loss_sum / sample_count
    if not math.isfinite(validation_loss):
        raise ValueError("validation loss must be finite")
    return validation_loss


def conditional_expectation_G_given_lmbda(model, likelihood, stat, *, batch_size=None):
    model.eval()
    likelihood.eval()
    if batch_size is None:
        batch_size = len(stat)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_device = _module_device(model, stat.device)
    means = []
    lowers = []
    uppers = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, len(stat), batch_size):
            batch = stat[start : start + batch_size].to(model_device)
            prediction = likelihood(model(batch))
            lower, upper = prediction.confidence_region()
            means.append(prediction.mean.detach().cpu())
            lowers.append(lower.detach().cpu())
            uppers.append(upper.detach().cpu())
    return torch.cat(means, dim=0), torch.cat(lowers, dim=0), torch.cat(uppers, dim=0)
