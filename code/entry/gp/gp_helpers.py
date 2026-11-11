from pathlib import Path
import subprocess

import numpy as np
import torch

from config.pause import cache_stem_fmt
from entry.gp.gp_identity import identity_for_fold, resolve_read_artifact
from models.gp import GPConfig, GPPipeline


def fold_name(profile, fold) -> str:
    return cache_stem_fmt(profile.pause_cfg).format(fold_id=fold.fold_id)


def select_target_folds(profile, all_folds, fold_ids):
    all_folds = tuple(all_folds)
    if fold_ids is None:
        return all_folds

    requested = [int(value) for value in fold_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("fold_ids must be unique canonical fold IDs")

    indexed = {
        identity_for_fold(profile.dataset, fold).canonical_fold_id: fold
        for fold in all_folds
    }
    unknown = sorted(set(requested) - set(indexed))
    if unknown:
        raise ValueError(f"fold_ids are not present in dataset: {unknown}")
    return tuple(indexed[value] for value in requested)


def build_gp_pipeline(
    profile,
    *,
    config: GPConfig,
    costs=None,
):
    config.num_classes = profile.num_classes

    if costs is not None:
        normalized_costs = [float(cost) for cost in costs]
        if (
            not normalized_costs
            or not np.isfinite(normalized_costs).all()
            or any(cost < 0 for cost in normalized_costs)
        ):
            raise ValueError("costs must be finite non-negative values")
        config.cost_pool = normalized_costs

    return GPPipeline(config)


def _extract_model_state(payload):
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    if isinstance(payload, dict) and "state" in payload:
        return payload["state"]
    return payload


def load_checkpoint_payload(profile, fold, *, device, strict):
    identity = identity_for_fold(profile.dataset, fold)
    try:
        checkpoint = resolve_read_artifact(
            profile.s1_checkpoint_dir,
            identity,
            ".pth",
        )
    except FileNotFoundError:
        if strict:
            raise
        return None, None
    payload = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    return payload, checkpoint


def load_checkpoint_state(profile, fold, *, device, logger):
    payload, checkpoint = load_checkpoint_payload(
        profile,
        fold,
        device=device,
        strict=False,
    )
    if payload is None:
        if logger is not None:
            logger.error(
                "[S2] missing S1 checkpoint for %s.pth",
                fold_name(profile, fold),
            )
        return None
    return _extract_model_state(payload)
