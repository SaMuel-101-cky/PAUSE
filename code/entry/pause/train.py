"""Cross-epoch PAUSE general training/evaluation engine (--dataset)."""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from config.pause import (
    MODEL_KWARGS, TRAIN_DEFAULTS, VARIANTS, get_cfg, output_dir as cfg_output_dir,
    ckpt_dir as cfg_ckpt_dir, cache_stem_fmt,
)
from dataset.folds import enumerate_folds
from dataset.pause import assemble_fold
from entry.pause.export import save_collected_outputs, save_history_tables
from entry.pause.history import RANDOM_HISTORY_END_IDX_DIST
from models.pause.model import resolve_rope_base
from training.core import log_summary_block
from training.pause import train_one_fold_pause


def mean_summary(blocks):
    def stack(key):
        return np.stack([block[key] for block in blocks])

    return {
        "acc": float(np.mean([block["acc"] for block in blocks])),
        "mf1": float(np.mean([block["mf1"] for block in blocks])),
        "kappa": float(np.mean([block["kappa"] for block in blocks])),
        "mgm": float(np.mean([block["mgm"] for block in blocks])),
        "precision_per_class": stack("precision_per_class").mean(0),
        "recall_per_class": stack("recall_per_class").mean(0),
        "f1_per_class": stack("f1_per_class").mean(0),
        "gmean_per_class": stack("gmean_per_class").mean(0),
        "confusion_matrix": stack("confusion_matrix").sum(0),
    }


LOG_VARIANT_LABELS = {name: spec.log_label for name, spec in VARIANTS.items()}


def build_logger(name, model_variant="paired"):
    if model_variant not in LOG_VARIANT_LABELS:
        raise ValueError(
            f"unknown model_variant '{model_variant}', "
            f"choices: {tuple(LOG_VARIANT_LABELS)}"
        )
    namespace = "pause" if model_variant == "paired" else f"pause_{model_variant}"
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(
        f"{namespace}_{name}_{time.time_ns()}"
    )
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler(
        log_dir / f"{namespace}_{name}_{time.strftime('%m-%d-%H%M%S')}.log"
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    ))
    logger.addHandler(handler)
    return logger


def _save_random_aggregate(path, aggregate):
    arrays = {}
    for branch, metrics in aggregate.items():
        for key, value in metrics.items():
            arrays[f"{branch}_{key}"] = np.asarray(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _save_full_curves(path, curves):
    arrays = {}
    for branch, metrics in curves.items():
        for key, value in metrics.items():
            arrays[f"{branch}_{key}"] = np.asarray(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _save_loso_summary(path, fold_ids, results, model_variant="paired"):
    if len(fold_ids) != len(results) or not results:
        raise ValueError("fold_ids and non-empty results must align")
    arrays = {
        "test_subject": np.asarray(fold_ids, dtype=np.int64),
        "model_variant": np.asarray(model_variant),
        "best_epoch": np.asarray(
            [result["best"]["epoch"] for result in results],
            dtype=np.int64,
        ),
        "best_ml_mean_mf1": np.asarray([
            result["best"]["curves"]["ml"]["mean_mf1"]
            for result in results
        ]),
    }
    curve_keys = (
        "acc_k",
        "mf1_k",
        "precision_per_k",
        "f1_per_k",
        "recall_per_k",
    )
    scalar_summary_keys = ("acc", "mf1", "kappa", "mgm")
    per_class_summary_keys = (
        "precision_per_class",
        "recall_per_class",
        "f1_per_class",
        "gmean_per_class",
    )
    random_seeds = tuple(sorted(results[0]["random_curves"]))
    for result in results[1:]:
        if tuple(sorted(result["random_curves"])) != random_seeds:
            raise ValueError("random-history seeds differ across folds")
    arrays["random_seed"] = np.asarray(random_seeds, dtype=np.int64)
    for branch in ("ml", "map"):
        for key in curve_keys:
            per_fold = np.stack([
                result["best"]["curves"][branch][key]
                for result in results
            ])
            arrays[f"full_{branch}_{key}_per_fold"] = per_fold
            arrays[f"full_{branch}_{key}_mean"] = per_fold.mean(axis=0)
            per_seed = np.stack([
                np.stack([
                    result["random_curves"][seed][branch][key]
                    for result in results
                ]).mean(axis=0)
                for seed in random_seeds
            ])
            arrays[f"random_{branch}_{key}_per_seed"] = per_seed
            arrays[f"random_{branch}_{key}_mean"] = per_seed.mean(axis=0)
            arrays[f"random_{branch}_{key}_std"] = per_seed.std(axis=0)
        for key in scalar_summary_keys + per_class_summary_keys:
            per_fold = np.stack([
                np.asarray(result["best"]["summary"][branch][key])
                for result in results
            ])
            arrays[f"full_{branch}_{key}_per_fold"] = per_fold
            arrays[f"full_{branch}_{key}_mean"] = per_fold.mean(axis=0)
        arrays[f"full_{branch}_confusion_matrix_sum"] = np.stack([
            result["best"]["summary"][branch]["confusion_matrix"]
            for result in results
        ]).sum(axis=0)
        random_mean_mf1 = np.stack([
            np.asarray([
                result["random_curves"][seed][branch]["mean_mf1"]
                for result in results
            ]).mean()
            for seed in random_seeds
        ])
        arrays[f"random_{branch}_mean_mf1_per_seed"] = random_mean_mf1
        arrays[f"random_{branch}_mean_mf1_mean"] = random_mean_mf1.mean()
        arrays[f"random_{branch}_mean_mf1_std"] = random_mean_mf1.std()
        for key in scalar_summary_keys + per_class_summary_keys:
            per_seed = np.stack([
                np.stack([
                    np.asarray(
                        result["random_summaries"][seed][branch][key]
                    )
                    for result in results
                ]).mean(axis=0)
                for seed in random_seeds
            ])
            arrays[f"random_{branch}_{key}_per_seed"] = per_seed
            arrays[f"random_{branch}_{key}_mean"] = per_seed.mean(axis=0)
            arrays[f"random_{branch}_{key}_std"] = per_seed.std(axis=0)
        random_confusion = np.stack([
            np.stack([
                result["random_summaries"][seed][branch][
                    "confusion_matrix"
                ]
                for result in results
            ]).sum(axis=0)
            for seed in random_seeds
        ])
        arrays[
            f"random_{branch}_confusion_matrix_per_seed"
        ] = random_confusion
        arrays[
            f"random_{branch}_confusion_matrix_mean"
        ] = random_confusion.mean(axis=0)
        arrays[
            f"random_{branch}_confusion_matrix_std"
        ] = random_confusion.std(axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _log_loso_text_summary(logger, results):
    if not results:
        return
    n_folds = len(results)
    ml_mf1_k30 = [
        result["best"]["curves"]["ml"]["mf1_k"][-1]
        for result in results
    ]
    map_mf1_k30 = [
        result["best"]["curves"]["map"]["mf1_k"][-1]
        for result in results
    ]
    ml_acc_k30 = [
        result["best"]["curves"]["ml"]["acc_k"][-1]
        for result in results
    ]
    map_acc_k30 = [
        result["best"]["curves"]["map"]["acc_k"][-1]
        for result in results
    ]
    logger.info(f"==== S1 summary (k=30 ceiling, {n_folds}-fold average) ====")
    logger.info(
        f"ML : acc@30={np.mean(ml_acc_k30):.4f}  "
        f"MF1@30={np.mean(ml_mf1_k30):.4f}"
    )
    logger.info(
        f"MAP: acc@30={np.mean(map_acc_k30):.4f}  "
        f"MF1@30={np.mean(map_mf1_k30):.4f}"
    )
    for tag, branch in (("ML", "ml"), ("MAP", "map")):
        summary = mean_summary([
            result["best"]["summary"][branch]
            for result in results
        ])
        logger.info(f"---- {tag} {n_folds}-fold average per-class metrics (k=30) ----")
        log_summary_block(logger, tag, summary)
        logger.info(
            f"  [{tag}] Confusion matrix ({n_folds}-fold sum, rows=true labels cols=predictions):\n"
            f"{summary['confusion_matrix']}"
        )


def run(
    cfg,
    *,
    output_dir=None,
    ckpt_dir=None,
    test_folds=None,
    num_train_epochs=None,
    num_window_epochs=None,
    batch_size_windows=None,
    device=None,
    model_kwargs=None,
    num_workers=4,
    early_stop_patience=None,
    variant="paired",
    train_one_fold_fn=None,
):
    """Run training and evaluation on all folds (or those specified by
    test_folds) for the dataset given by cfg."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant '{variant}', choices: {tuple(VARIANTS)}")
    spec = VARIANTS[variant]
    train_history = spec.train_history
    if model_kwargs is None:
        model_kwargs = dict(MODEL_KWARGS)
    if not spec.cross_epoch_memory:
        model_kwargs = {**model_kwargs, "cross_epoch_memory": False}
    num_train_epochs = num_train_epochs or TRAIN_DEFAULTS["num_train_epochs"]
    num_window_epochs = num_window_epochs or TRAIN_DEFAULTS["num_window_epochs"]
    early_stop_patience = early_stop_patience or TRAIN_DEFAULTS["early_stop_patience"]
    batch_size_windows = batch_size_windows or cfg.batch_size_windows
    output_dir = Path(output_dir or cfg_output_dir(cfg, variant))
    ckpt_dir = Path(ckpt_dir or cfg_ckpt_dir(cfg, variant))
    logger = build_logger(cfg.name, variant)
    logger.info(
        "==== PAUSE training variant: %s (model_variant=%s) ====",
        LOG_VARIANT_LABELS[variant],
        variant,
    )
    logger.info("Evaluation protocol: model selection=H-Full, export=H-Full+H-Random")
    if train_one_fold_fn is None:
        train_one_fold_fn = train_one_fold_pause

    folds = enumerate_folds(cfg)
    if test_folds is not None:
        want = set(test_folds)
        folds = [f for f in folds if f.fold_id in want]
    results, fold_ids = [], []
    for fold in folds:
        stem = cache_stem_fmt(cfg).format(fold_id=fold.fold_id)
        pf = assemble_fold(
            fold, num_epochs=num_window_epochs, num_classes=TRAIN_DEFAULTS["num_classes"]
        )
        logger.info(
            "%s: train %s test %s M_k=%s",
            stem, pf.train.x.shape, pf.test.x.shape,
            list(pf.class_counts),
        )
        result = train_one_fold_fn(
            pf,
            num_train_epochs=num_train_epochs,
            batch_size_windows=batch_size_windows,
            device=device,
            model_kwargs=model_kwargs,
            lr=TRAIN_DEFAULTS["lr"],
            weight_decay=TRAIN_DEFAULTS["weight_decay"],
            min_lr=TRAIN_DEFAULTS["min_lr"],
            grad_clip=TRAIN_DEFAULTS["grad_clip"],
            seed=TRAIN_DEFAULTS["seed"],
            num_workers=num_workers,
            early_stop_patience=early_stop_patience,
            logger=logger,
            progress_desc=stem,
            train_history=train_history,
        )
        best = result["best"]
        checkpoint = ckpt_dir / cfg.ckpt_name_fmt.format(fold_id=fold.fold_id)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_model_kwargs = {
            **model_kwargs,
            "num_epochs": num_window_epochs,
            "num_patches": int(pf.train.x.shape[2]),
            "num_classes": TRAIN_DEFAULTS["num_classes"],
            "rope_base": resolve_rope_base(
                model_kwargs.get("rope_mode", "2d"), model_kwargs.get("rope_base")
            ),
        }
        torch.save({
            "model_state": best["state"],
            "model_kwargs": checkpoint_model_kwargs,
            "train_mean": pf.train_mean,
            "train_std": pf.train_std,
            "fold_id": fold.fold_id,
            "test_units": list(fold.test_units),
            "test_subject": fold.fold_id if cfg.fold_scheme == "loso" else -1,
            "best_epoch": best["epoch"],
            "best_ml_mean_mf1": best["mean_mf1"],
            "selection_history_mode": "full",
            "train_history_mode": train_history,
            "random_history_end_idx_dist": RANDOM_HISTORY_END_IDX_DIST,
            "model_variant": variant,
        }, checkpoint)
        _save_full_curves(output_dir / "curves" / f"{stem}_full.npz", best["curves"])
        save_history_tables(
            output_dir / "history_tables" / f"{stem}.npz",
            result["history_tables"],
        )
        save_collected_outputs(
            output_dir / "exports" / f"{stem}_full.npz",
            best["full_outputs"], best["log_prior"], history_mode="full",
        )
        for random_seed, collected in result["random_outputs"].items():
            save_collected_outputs(
                output_dir / "exports" / f"{stem}_random_seed{random_seed}.npz",
                collected, best["log_prior"],
                history_mode=f"random_seed_{random_seed}", random_seed=random_seed,
            )
        _save_random_aggregate(
            output_dir / "curves" / f"{stem}_random.npz",
            result["random_aggregate"],
        )
        logger.info(
            "%s best@ep%s by meanMF1=%.4f (per-class detail k=30):",
            stem,
            best["epoch"],
            best["mean_mf1"],
        )
        log_summary_block(logger, "ML", best["summary"]["ml"])
        log_summary_block(logger, "MAP", best["summary"]["map"])
        results.append(result)
        fold_ids.append(fold.fold_id)
    _save_loso_summary(output_dir / "loso_summary.npz", fold_ids, results, variant)
    _log_loso_text_summary(logger, results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="PAUSE cross-epoch training (--dataset)")
    parser.add_argument("--dataset", required=True, choices=["dreams", "edf20", "edf78", "shhs"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--variant", choices=tuple(VARIANTS), default="paired",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_cfg(args.dataset)
    if args.smoke:
        folds = enumerate_folds(cfg)
        run(cfg, test_folds=[folds[0].fold_id], num_train_epochs=1,
            device=args.device, num_workers=args.workers,
            variant=args.variant)
    else:
        run(cfg, device=args.device, num_workers=args.workers,
            variant=args.variant)
    print("Finished!")


if __name__ == "__main__":
    main()
