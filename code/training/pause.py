"""PAUSE cross-epoch classifier training."""
import copy
import math

import torch
import torch.nn.functional as F
from lion_pytorch import Lion
from tqdm import tqdm

from entry.pause.eval import (
    aggregate_random_curves,
    collect_outputs,
    evaluate_collected,
    evaluate_selection,
    evaluate_summary,
)
from entry.pause.history import (
    RANDOM_HISTORY_SEEDS,
    build_random_history_tables,
    sample_training_end_idx,
)
from models.pause.model import PAUSE
from training.core import (
    CosineAnnealingWarmupRestarts,
    log_summary_block,
    set_seed,
)
from dataset.core import class_weights_from_counts, priors_from_counts
from dataset.pause import make_window_loaders


def build_paired_history_batch(x, labels):
    if x.ndim != 4:
        raise ValueError("x must be [B,E,P,F]")
    batch, epochs, patches, _ = x.shape
    if labels.shape != (batch, epochs):
        raise ValueError("labels must be [B,E]")
    random_end_idx = sample_training_end_idx(
        batch,
        epochs,
        patches,
        x.device,
    )
    full_end_idx = torch.full_like(random_end_idx, patches - 1)
    return (
        torch.cat((x, x), dim=0),
        torch.cat((labels, labels), dim=0),
        torch.cat((full_end_idx, random_end_idx), dim=0),
    )


def build_full_history_batch(x, labels):
    if x.ndim != 4:
        raise ValueError("x must be [B,E,P,F]")
    batch, epochs, patches, _ = x.shape
    if labels.shape != (batch, epochs):
        raise ValueError("labels must be [B,E]")
    full_end_idx = torch.full(
        (batch, epochs), patches - 1, dtype=torch.long, device=x.device
    )
    return x, labels, full_end_idx


def build_random_history_batch(x, labels):
    if x.ndim != 4:
        raise ValueError("x must be [B,E,P,F]")
    batch, epochs, patches, _ = x.shape
    if labels.shape != (batch, epochs):
        raise ValueError("labels must be [B,E]")
    random_end_idx = sample_training_end_idx(batch, epochs, patches, x.device)
    return x, labels, random_end_idx


def select_history_builder(train_history):
    if train_history == "paired":
        return build_paired_history_batch
    if train_history == "full":
        return build_full_history_batch
    if train_history == "random":
        return build_random_history_batch
    raise ValueError(
        f"unknown train_history '{train_history}', choices: paired/full/random"
    )


def lsel_loss_pause(logits, labels, class_weight):
    if logits.ndim != 4:
        raise ValueError("logits must be [B,E,P,K]")
    batch, epochs, patches, classes = logits.shape
    if labels.shape != (batch, epochs):
        raise ValueError("labels must be [B,E]")
    targets = labels[:, :, None].expand(
        batch, epochs, patches
    ).reshape(-1)
    return F.cross_entropy(
        logits.reshape(batch * epochs * patches, classes),
        targets,
        weight=class_weight,
        reduction="mean",
    )


def train_one_fold_pause(
    fold,
    num_train_epochs,
    batch_size_windows,
    device,
    model_kwargs=None,
    lr=1e-4,
    weight_decay=0.01,
    warmup_epochs=None,
    min_lr=1e-6,
    seed=2025,
    num_workers=4,
    num_classes=5,
    logger=None,
    sched_epochs=None,
    grad_clip=5.0,
    early_stop_patience=None,
    progress_desc=None,
    evaluate_random_histories=True,
    train_history="paired",
):
    set_seed(seed)
    history_builder = select_history_builder(train_history)
    if sched_epochs is None:
        sched_epochs = num_train_epochs
    if not sched_epochs >= num_train_epochs >= 1:
        raise ValueError("sched_epochs must be >= num_train_epochs >= 1")
    if early_stop_patience is not None and (
        not isinstance(early_stop_patience, int)
        or isinstance(early_stop_patience, bool)
        or early_stop_patience < 1
    ):
        raise ValueError("early_stop_patience must be a positive int or None")
    train_loader, test_loader = make_window_loaders(
        fold.train,
        fold.test,
        batch_size_windows,
        seed=seed,
        num_workers=num_workers,
    )
    class_weight = class_weights_from_counts(fold.class_counts).to(device)
    log_prior = torch.log(priors_from_counts(fold.class_counts)).to(device)
    epochs = fold.train.x.shape[1]
    patches = fold.train.x.shape[2]
    model_kwargs = dict(model_kwargs or {})
    reserved = {"num_epochs", "num_patches", "num_classes"}
    overlap = reserved.intersection(model_kwargs)
    if overlap:
        raise ValueError(f"model_kwargs contains reserved keys: {overlap}")
    model = PAUSE(
        num_epochs=epochs,
        num_patches=patches,
        num_classes=num_classes,
        **model_kwargs,
    ).to(device)
    optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
    if warmup_epochs is None:
        warmup_epochs = max(1, sched_epochs // 10)
    warmup_epochs = min(warmup_epochs, max(0, sched_epochs - 1))
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=sched_epochs,
        warmup_steps=warmup_epochs,
        max_lr=lr,
        min_lr=min_lr,
    )
    best = {
        "mean_mf1": -1.0,
        "epoch": -1,
        "curves": None,
        "summary": None,
        "state": None,
    }
    use_bar = progress_desc is not None
    epoch_iter = (
        tqdm(
            range(num_train_epochs),
            desc=progress_desc,
            dynamic_ncols=True,
            leave=True,
        )
        if use_bar
        else range(num_train_epochs)
    )
    epochs_since_best = 0   # consecutive epochs without a new best (early-stopping counter)
    for train_epoch in epoch_iter:
        model.train()
        detached_losses = []
        for batch in train_loader:
            x = batch["x"].float().to(device, non_blocking=True)
            y = batch["y"].long().to(device, non_blocking=True)
            batch_x, batch_y, batch_end_idx = history_builder(x, y)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_end_idx)
            loss = lsel_loss_pause(logits, batch_y, class_weight)
            loss.backward()
            if grad_clip is not None and math.isfinite(grad_clip):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )
            optimizer.step()
            detached_losses.append(loss.detach())
        loss_values = torch.stack(detached_losses).cpu().tolist()
        total_loss = sum(loss_values)
        batches = len(loss_values)
        scheduler.step()
        collected = collect_outputs(
            model,
            test_loader,
            device,
            num_patches=patches,
            history_table=None,
            return_summary=False,
        )
        # Per-epoch we only compute the ML per-k macro-F1 needed for best selection
        # (pointwise identical to the full evaluation); full curves / per-class
        # summaries are "compute-and-discard" intermediate layers — they are not
        # recomputed every epoch, only once at fold-end on the best weights (see
        # after the loop). The summary is computed on demand only when a logger is
        # available — the HPO path (logger=None) skips the entire block.
        ml_curve = evaluate_selection(collected, num_classes=num_classes)
        score = float(ml_curve["mean_mf1"])
        if logger is not None:
            epoch_summary = evaluate_summary(
                collected,
                log_prior.cpu(),
                num_classes=num_classes,
            )
            logger.info(
                "ep %s/%s loss=%.4f  (k=30 per-class metrics↓)",
                train_epoch + 1,
                num_train_epochs,
                total_loss / max(batches, 1),
            )
            log_summary_block(logger, "ML", epoch_summary["ml"], num_classes)
            log_summary_block(logger, "MAP", epoch_summary["map"], num_classes)
        if score > best["mean_mf1"]:
            best.update(
                mean_mf1=score,
                epoch=train_epoch + 1,
                state=copy.deepcopy(model.state_dict()),
            )
            epochs_since_best = 0
            if use_bar:
                tqdm.write(
                    f"★ {progress_desc} new best @ep{train_epoch + 1}: "
                    f"ML meanMF1={score:.4f} "
                    f"(MF1@30={ml_curve['mf1_at_last']:.4f} "
                    f"acc={ml_curve['acc_at_last']:.4f})"
                )
        else:
            epochs_since_best += 1
        if use_bar:
            epoch_iter.set_postfix(
                loss=f"{total_loss / max(batches, 1):.3f}",
                best_meanmf1=f"{best['mean_mf1']:.3f}",
            )
        # early_stop_patience consecutive epochs without a new best → end this fold early.
        if (early_stop_patience is not None
                and epochs_since_best >= early_stop_patience):
            if logger is not None:
                logger.info(
                    "early stop @ep%s (best @ep%s, patience=%s)",
                    train_epoch + 1, best["epoch"], early_stop_patience,
                )
            if use_bar:
                tqdm.write(
                    f"■ {progress_desc} early stop @ep{train_epoch + 1} "
                    f"(best @ep{best['epoch']}, patience={early_stop_patience})"
                )
            break
    best["epochs_run"] = train_epoch + 1   # actual number of epochs trained (less than num_train_epochs when early-stopped)
    model.load_state_dict(best["state"], strict=True)
    full_outputs = collect_outputs(
        model,
        test_loader,
        device,
        num_patches=patches,
        history_table=None,
        return_summary=True,
    )
    full_curves, full_summary = evaluate_collected(
        full_outputs,
        log_prior.cpu(),
        num_classes=num_classes,
    )
    best.update(
        curves=full_curves,
        summary=full_summary,
        full_outputs=full_outputs,
        log_prior=log_prior.cpu(),
    )
    if not evaluate_random_histories:
        return {
            "best": best,
            "history_tables": {},
            "random_curves": {},
            "random_summaries": {},
            "random_outputs": {},
            "random_aggregate": None,
        }
    history_tables = build_random_history_tables(
        num_windows=len(fold.test.x),
        num_epochs=epochs,
        num_patches=patches,
        seeds=RANDOM_HISTORY_SEEDS,
    )
    random_curves = {}
    random_summaries = {}
    random_outputs = {}
    for random_seed, table in history_tables.items():
        collected = collect_outputs(
            model,
            test_loader,
            device,
            num_patches=patches,
            history_table=table,
            return_summary=True,
        )
        curves, summary = evaluate_collected(
            collected,
            log_prior.cpu(),
            num_classes=num_classes,
        )
        random_curves[random_seed] = curves
        random_summaries[random_seed] = summary
        random_outputs[random_seed] = collected
    return {
        "best": best,
        "history_tables": history_tables,
        "random_curves": random_curves,
        "random_summaries": random_summaries,
        "random_outputs": random_outputs,
        "random_aggregate": aggregate_random_curves(
            random_curves,
            random_summaries,
        ),
    }
