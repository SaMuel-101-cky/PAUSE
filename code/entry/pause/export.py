"""Cross-epoch classifier result export."""
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".npz",
    )
    os.close(fd)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_collected_outputs(
    path,
    collected,
    log_prior,
    history_mode,
    random_seed=None,
):
    logits = collected.logits.float().cpu()
    log_prior = torch.as_tensor(log_prior).float().cpu()
    arrays = {
        "logits": logits.numpy(),
        "posterior_ml": torch.softmax(logits, dim=-1).numpy(),
        "posterior_map": torch.softmax(
            logits + log_prior.reshape(1, 1, 1, -1),
            dim=-1,
        ).numpy(),
        "posterior_temperature": np.asarray(1.0, dtype=np.float32),
        "label": collected.labels.numpy(),
        "end_idx": collected.end_idx.numpy(),
        "subject_id": collected.subject_id.numpy(),
        "window_id": collected.window_id.numpy(),
        "source_epoch_index": collected.source_epoch_index.numpy(),
        "occurrence_index": collected.occurrence_index.numpy(),
        "epoch_position": collected.epoch_position.numpy(),
        "log_prior": log_prior.numpy(),
        "history_mode": np.asarray(history_mode),
        "random_seed": np.asarray(
            -1 if random_seed is None else int(random_seed),
            dtype=np.int64,
        ),
    }
    if collected.context_summary is not None:
        arrays["context_summary"] = collected.context_summary.numpy()
    _atomic_savez(path, **arrays)


def save_history_tables(path, tables):
    arrays = {
        f"seed_{int(seed)}": table.cpu().numpy()
        for seed, table in sorted(tables.items())
    }
    _atomic_savez(path, **arrays)


def load_history_tables(path):
    with np.load(path, allow_pickle=False) as saved:
        tables = {}
        for name in saved.files:
            if not name.startswith("seed_"):
                raise ValueError(f"unexpected history table key {name}")
            seed = int(name.removeprefix("seed_"))
            tables[seed] = torch.from_numpy(saved[name].copy()).long()
    return dict(sorted(tables.items()))
