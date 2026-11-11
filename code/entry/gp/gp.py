import argparse
import logging
import os
import time

from config.gp import GPConfig
from config.pause import VARIANTS
from dataset.folds import enumerate_folds
from entry.gp.gp_helpers import (
    build_gp_pipeline,
    fold_name,
    select_target_folds,
)
from entry.gp.gp_identity import identity_for_fold
from entry.gp.pause_backend import PauseGPBackend
from entry.gp.gp_profiles import get_gp_profile
from training.core import set_seed


def build_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"gp_entry_{time.time_ns()}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler(
        os.path.join(log_dir, f"gp_entry_{time.strftime('%m-%d-%H%M%S')}.log")
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def make_gp_run_dir(profile):
    run_id = time.strftime("%m-%d-%H%M%S")
    checkpoint_run_dir = os.path.join(profile.gp_checkpoint_dir, run_id)
    output_run_dir = os.path.join(profile.gp_output_dir, run_id)
    os.makedirs(checkpoint_run_dir, exist_ok=True)
    os.makedirs(output_run_dir, exist_ok=True)
    return checkpoint_run_dir, output_run_dir


def run_gp(
    dataset,
    *,
    variant,
    smoke=False,
    device=None,
    posterior_device=None,
    penalty_alpha=None,
    workers=None,
    fold_ids=None,
    costs=None,
):
    config = GPConfig()
    if device is not None:
        config.device = device
    if posterior_device is not None:
        config.posterior_device = posterior_device
    if penalty_alpha is not None:
        config.penalty_alpha = penalty_alpha
    if workers is not None:
        config.num_workers = workers
    set_seed(config.seed)

    logger = build_logger(config.log_dir)
    profile = get_gp_profile(dataset, variant)
    backend = PauseGPBackend()
    pipeline = build_gp_pipeline(
        profile,
        config=config,
        costs=costs,
    )
    all_folds = enumerate_folds(profile.pause_cfg)
    selected_folds = select_target_folds(profile, all_folds, fold_ids)
    selected_fold_ids = tuple(
        identity_for_fold(profile.dataset, fold).canonical_fold_id
        for fold in selected_folds
    )
    checkpoint_run_dir, output_run_dir = make_gp_run_dir(profile)
    logger.info("[S2] GP checkpoints: %s", checkpoint_run_dir)
    logger.info("[S2] GP outputs: %s", output_run_dir)
    all_gp_results = {}

    for fold in selected_folds:
        name = fold_name(profile, fold)
        result = backend.run_fold(
            profile=profile,
            fold=fold,
            all_folds=all_folds,
            pipeline=pipeline,
            logger=logger,
            smoke=smoke,
            checkpoint_run_dir=checkpoint_run_dir,
            output_run_dir=output_run_dir,
            selected_fold_ids=selected_fold_ids,
        )
        if result is None:
            continue
        all_gp_results[name] = result
        if smoke:
            break

    logger.info(
        "[S2] completed folds: %d/%d",
        len(all_gp_results),
        len(selected_folds),
    )
    if all_gp_results:
        pipeline.print_summary(
            all_gp_results,
            logger,
            output_run_dir=output_run_dir,
            expected_folds=len(selected_folds),
        )
    return all_gp_results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PAUSE GP backward induction")
    parser.add_argument("--dataset", default="dreams")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--device", default=None
    )
    parser.add_argument(
        "--posterior-device",
        choices=("cpu", "gpu"),
        default=None,
        help="Where GP training posteriors are retained: cpu is OOM-safe, gpu is faster on small datasets.",
    )
    parser.add_argument(
        "--penalty-alpha",
        type=float,
        default=None,
        help="Override GP class-imbalance penalty alpha for this run.",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--fold_ids", type=int, nargs="+", default=None)
    parser.add_argument("--costs", type=float, nargs="+", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_gp(
        args.dataset,
        smoke=args.smoke,
        device=args.device,
        posterior_device=args.posterior_device,
        penalty_alpha=args.penalty_alpha,
        workers=args.workers,
        variant=args.variant,
        fold_ids=args.fold_ids,
        costs=args.costs,
    )
    print("Finished!")


if __name__ == "__main__":
    main()
