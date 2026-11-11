import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger


def save_gaussian_process_models(
    model_pool, model_path, likelihood_pool, likelihood_path, is_entire_model=True
):
    """
    Args:
    - model_pool: a list of length time_steps - 1, containing GPyTorch models.
    - likelihood_pool: a list of length time_steps - 1, containing GPyTorch likelihoods.
    """
    if is_entire_model:
        torch.save(model_pool, model_path)
        torch.save(likelihood_pool, likelihood_path)
    else:
        state_dicts = [model.state_dict() for model in model_pool]
        torch.save(state_dicts, model_path)

        state_dicts = [likelihood.state_dict() for likelihood in likelihood_pool]
        torch.save(state_dicts, likelihood_path)

    logger.success("Successfully saved GPyTorch models and Likelihoods")

def load_gaussian_process_models(model_path, likelihood_path):
    """
    Args:
    - model_pool: a list of length time_steps - 1, containing GPyTorch models.
    - likelihood_pool: a list of length time_steps - 1, containing GPyTorch likelihoods.
    """

    model = torch.load(model_path)
    likelihood = torch.load(likelihood_path)

    logger.success("Successfully loaded GPyTorch models and Likelihoods")

    return model, likelihood

def save_gp_model(
    model_pool,
    likelihood_pool,
    continuation_risk,
    stopping_risk,
    predicted_labels,
    loss_pool,
    model_dirname,
    artifact_dirname,
    *,
    epoch_train_loss_pool=None,
    epoch_validation_loss_pool=(),
    training_report=None,
    max_epochs=None,
):
    model_dirname = str(model_dirname)
    artifact_dirname = str(artifact_dirname)
    save_gaussian_process_models(
        model_pool,
        model_dirname + "/GPmodels.pt",
        likelihood_pool,
        model_dirname + "/Likelihoodmodels.pt",
    )
    np.save(artifact_dirname + "/continuation_risk", continuation_risk.cpu())
    np.save(artifact_dirname + "/stopping_risk", stopping_risk.cpu())
    np.save(artifact_dirname + "/predicted_labels", predicted_labels.cpu())
    if training_report is None:
        training_report = {
            "early_stopping_enabled": False,
            "time_steps": [],
        }
    if epoch_train_loss_pool is None:
        epoch_train_loss_pool = loss_pool
    save_gp_training_artifacts(
        batch_histories=loss_pool,
        epoch_train_histories=epoch_train_loss_pool,
        epoch_validation_histories=epoch_validation_loss_pool,
        report=training_report,
        artifact_dirname=artifact_dirname,
        max_epochs=max_epochs,
    )


def _pad_histories(histories, max_length=None) -> np.ndarray:
    rows = [np.asarray(history, dtype=np.float64).reshape(-1) for history in histories]
    observed_max = max((len(row) for row in rows), default=0)
    if max_length is None:
        max_len = observed_max
    else:
        max_len = int(max_length)
        if max_len < observed_max:
            raise ValueError(
                "max_length must be at least the longest history length"
            )
    padded = np.full((len(rows), max_len), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        padded[index, : len(row)] = row
    return padded


def save_gp_training_artifacts(
    *,
    batch_histories,
    epoch_train_histories,
    epoch_validation_histories,
    report,
    artifact_dirname,
    max_epochs=None,
):
    artifact_dir = Path(artifact_dirname)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    batch_array = _pad_histories(batch_histories)
    train_array = _pad_histories(epoch_train_histories, max_epochs)
    validation_array = _pad_histories(
        epoch_validation_histories,
        max_epochs if report.get("early_stopping_enabled", False) else None,
    )
    np.save(artifact_dir / "loss_pool.npy", batch_array)
    np.save(artifact_dir / "epoch_train_loss.npy", train_array)
    np.save(artifact_dir / "epoch_validation_loss.npy", validation_array)

    with (artifact_dir / "early_stopping_metadata.json").open(
        "w", encoding="utf-8"
    ) as metadata_file:
        json.dump(report, metadata_file, indent=2)

    plotted = (
        validation_array
        if report.get("early_stopping_enabled", False)
        else train_array
    )
    figure, axes = plt.subplots()
    masked = np.ma.masked_invalid(plotted)
    if plotted.shape[1] > 0:
        axes.plot(masked.T)
    axes.set_ylabel("negative ELBO")
    axes.set_xlabel("epoch")
    figure.savefig(artifact_dir / "Negative_ELBO.pdf")
    plt.close(figure)
