from copy import deepcopy
from dataclasses import dataclass

import torch

from .costs import as_cost_matrix


@dataclass(frozen=True)
class GPLogitShard:
    canonical_fold_id: int
    logits: torch.Tensor
    labels: torch.Tensor
    log_prior: torch.Tensor
    oof_path: str | None = None
    oof_sha256: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    data_fingerprint: str | None = None

    def __post_init__(self):
        canonical_fold_id = int(self.canonical_fold_id)
        logits = torch.as_tensor(
            self.logits, dtype=torch.float32, device="cpu"
        ).detach()
        labels = torch.as_tensor(
            self.labels, dtype=torch.long, device="cpu"
        ).detach()
        log_prior = torch.as_tensor(
            self.log_prior, dtype=torch.float32, device="cpu"
        ).detach()

        if canonical_fold_id < 0:
            raise ValueError("canonical_fold_id must be non-negative")
        if logits.ndim != 3:
            raise ValueError("logits must be [N,T,K]")
        if labels.shape != (logits.shape[0],):
            raise ValueError("labels must match logits rows")
        if log_prior.shape != (logits.shape[2],):
            raise ValueError("log_prior must be [K]")
        if not torch.isfinite(logits).all():
            raise ValueError("GP logit shard contains NaN or Inf")
        if not torch.isfinite(log_prior).all():
            raise ValueError("log_prior contains NaN or Inf")
        if not torch.allclose(
            log_prior.exp().sum(),
            log_prior.new_tensor(1.0),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("log_prior probabilities must sum to one")
        if torch.any((labels < 0) | (labels >= logits.shape[2])):
            raise ValueError("labels must be in [0, K)")

        object.__setattr__(self, "canonical_fold_id", canonical_fold_id)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "log_prior", log_prior)
        for name in (
            "oof_path",
            "oof_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "data_fingerprint",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else str(value))


@dataclass(frozen=True)
class LoadedTemperature:
    temperature: torch.Tensor | None
    source_fold_ids: tuple[int, ...]
    path: str | None
    sha256: str | None

    def __post_init__(self):
        temperature = self.temperature
        if temperature is not None:
            temperature = torch.as_tensor(
                temperature, dtype=torch.float32, device="cpu"
            ).detach()
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(
            self,
            "source_fold_ids",
            tuple(int(fold_id) for fold_id in self.source_fold_ids),
        )


@dataclass(frozen=True)
class GPTrainingStates:
    llrs: torch.Tensor | None
    posteriors: torch.Tensor
    labels: torch.Tensor
    device: str | torch.device = "cpu"

    def __post_init__(self):
        device = torch.device(self.device)
        posteriors = torch.as_tensor(self.posteriors, device=device).detach()
        labels = torch.as_tensor(
            self.labels, dtype=torch.long, device="cpu"
        ).detach()
        if posteriors.ndim != 3:
            raise ValueError("posteriors must be [N,T,K]")
        if labels.shape != (posteriors.shape[0],):
            raise ValueError("labels must match training state rows")
        llrs = self.llrs
        if llrs is not None:
            llrs = torch.as_tensor(llrs, device=device).detach()
            if llrs.ndim != 4 or llrs.shape[2] != llrs.shape[3]:
                raise ValueError("llrs must be [N,T,K,K]")
            if posteriors.shape != llrs.shape[:3]:
                raise ValueError("posteriors must match LLR state [N,T,K]")
        object.__setattr__(self, "llrs", llrs)
        object.__setattr__(self, "posteriors", posteriors)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "device", str(device))


@dataclass(frozen=True)
class GPPolicy:
    model_pool: tuple
    likelihood_pool: tuple
    cost: float
    upper_horizon: int | None
    cost_matrix: torch.Tensor
    cost_metadata: dict
    training_report: dict | None = None

    def __post_init__(self):
        model_pool = tuple(self.model_pool)
        likelihood_pool = tuple(self.likelihood_pool)
        if len(model_pool) != len(likelihood_pool):
            raise ValueError("model and likelihood pools must have equal lengths")
        cost_matrix = torch.as_tensor(
            self.cost_matrix,
            dtype=torch.float32,
            device="cpu",
        ).detach().clone()
        if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ValueError("cost_matrix must be square [K,K]")
        cost_matrix = as_cost_matrix(
            cost_matrix,
            cost_matrix.shape[0],
            device="cpu",
            dtype=torch.float32,
        )
        if not isinstance(self.cost_metadata, dict):
            raise ValueError("cost_metadata must be a dictionary")
        object.__setattr__(self, "model_pool", model_pool)
        object.__setattr__(self, "likelihood_pool", likelihood_pool)
        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(self, "cost_matrix", cost_matrix)
        object.__setattr__(self, "cost_metadata", deepcopy(self.cost_metadata))
        object.__setattr__(
            self,
            "training_report",
            None if self.training_report is None else dict(self.training_report),
        )
