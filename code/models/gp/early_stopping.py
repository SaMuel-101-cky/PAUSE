import copy
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ValidationSplit:
    train_indices: torch.Tensor
    validation_indices: torch.Tensor


def make_validation_split(num_samples, fraction, seed, device) -> ValidationSplit:
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")

    validation_size = min(
        num_samples - 1,
        max(1, round(num_samples * fraction)),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)

    return ValidationSplit(
        train_indices=permutation[validation_size:].to(device),
        validation_indices=permutation[:validation_size].to(device),
    )


class EarlyStoppingController:
    def __init__(
        self,
        min_epochs,
        patience,
        min_delta,
        lr_reduce_factor,
        max_lr_reductions,
    ):
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.lr_reduce_factor = lr_reduce_factor
        self.max_lr_reductions = max_lr_reductions

        self.best_validation_loss = None
        self.best_epoch = None
        self.bad_epochs = 0
        self.lr_reductions = 0

        self._best_model_state = None
        self._best_likelihood_state = None
        self._best_optimizer_state = None

    def observe(
        self,
        epoch,
        validation_loss,
        model,
        likelihood,
        optimizer,
        *,
        allow_lr_reduction=True,
    ) -> str:
        validation_loss = float(validation_loss)
        if not math.isfinite(validation_loss):
            raise ValueError("validation_loss must be finite")

        if (
            self.best_validation_loss is None
            or self.best_validation_loss - validation_loss > self.min_delta
        ):
            self.best_validation_loss = validation_loss
            self.best_epoch = epoch
            self.bad_epochs = 0
            self._best_model_state = copy.deepcopy(model.state_dict())
            self._best_likelihood_state = copy.deepcopy(likelihood.state_dict())
            self._best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            return "improved"

        if epoch < self.min_epochs:
            return "continue"

        self.bad_epochs += 1
        if self.bad_epochs < self.patience:
            return "continue"

        if not allow_lr_reduction:
            return "continue"

        if self.lr_reductions >= self.max_lr_reductions:
            return "stop"

        self.restore_best(model, likelihood, optimizer)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] *= self.lr_reduce_factor
        self.lr_reductions += 1
        self.bad_epochs = 0
        return "reduce_lr"

    def restore_best(self, model, likelihood, optimizer) -> None:
        if self._best_model_state is None:
            raise RuntimeError("no best state is available")

        model.load_state_dict(copy.deepcopy(self._best_model_state))
        likelihood.load_state_dict(copy.deepcopy(self._best_likelihood_state))
        optimizer.load_state_dict(copy.deepcopy(self._best_optimizer_state))
