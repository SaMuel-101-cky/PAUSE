"""
Convert SHHS raw .mat files into the standard format shared with the other three datasets.

Input:  dataset/shhs1/*.mat   — raw SHHS (125Hz)
    EEG    (n, 3750)  float64  — 30 s / epoch, 125Hz
    label  (1, n)     int32    — 5-class {0..4}

Output: dataset/SHHS/<same_name>.mat  — three-key format identical to EDF78/EDF20/DREAMS
    eeg     (n, 1, 3000)  float32  — downsampled 125→100Hz then unsqueeze(1)
    label   (n, 1)        float32  — reshaped from (1, n) to column vector
    PSD100  (n, 30, 128)  float32  — STFT magnitude spectrum (dB), bins 1:129

Processing notes:
  1) SHHS raw signal is 125Hz; use resample_poly(up=4, down=5) to downsample to 100Hz (3750→3000).
  2) Key names / dimensions / dtype are unified with other datasets (PSD100, not the old script's PSD).
  3) Skip macOS ._* resource fork files to avoid loadmat crashes.

Safety policy: default dry-run (read-only, print shapes of first few files); pass --apply to actually write.
Writes go through a temp file then os.replace for atomic replacement.
"""
import os
import sys
import argparse
import numpy as np
import torch
from scipy.io import loadmat, savemat
from scipy.signal import resample_poly

# Add parent directory to Python path for importing dataset modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.stft import STFT

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE, 'shhs1')   # raw SHHS (keep unchanged)
DST_DIR = os.path.join(BASE, 'SHHS')    # standardized output

EXPECT_EEG_TAIL = (1, 3000)
EXPECT_PSD_TAIL = (30, 128)


def pick_device():
    """Prefer MPS, fall back to CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def real_mat_files(folder):
    """Return real .mat filenames (skip macOS ._* resource fork files)."""
    return [f for f in sorted(os.listdir(folder))
            if f.endswith('.mat') and not f.startswith('._')]


def transform(m, stft, device):
    """Build a standardized dict from a raw SHHS mat dict, with validation. Returns (new_dict, note)."""
    # 1) Downsample: 125Hz -> 100Hz, up=4/down=5 => 3750 -> 3000
    eeg_raw = np.asarray(m['EEG'])              # (n, 3750)
    n = eeg_raw.shape[0]
    eeg_ds = resample_poly(eeg_raw, up=4, down=5, axis=1)  # (n, 3000)

    # 2) STFT: input (n, 1, 3000) -> magnitude_db (n, 30, 128)
    eeg_t = torch.tensor(eeg_ds).unsqueeze(1).float().to(device)  # (n, 1, 3000)
    with torch.no_grad():
        _, _, _, _, _, magnitude_db = stft(eeg_t)

    eeg = eeg_t.cpu().numpy().astype(np.float32)              # (n, 1, 3000)
    psd = magnitude_db.cpu().numpy().astype(np.float32)       # (n, 30, 128)

    # 3) Unify label as (n, 1) column vector
    label = np.asarray(m['label']).reshape(-1, 1).astype(np.float32)

    # ---- Validation ----
    assert eeg.shape[1:] == EXPECT_EEG_TAIL, f"eeg trailing dims unexpected: {eeg.shape}"
    assert psd.shape[1:] == EXPECT_PSD_TAIL, f"PSD100 trailing dims unexpected: {psd.shape}"
    assert psd.shape[0] == n, f"PSD100 epoch count {psd.shape[0]} != eeg {n}"
    assert label.shape == (n, 1), f"label shape unexpected: {label.shape}"
    uniq = np.unique(label)
    assert np.all(np.isin(uniq, [0, 1, 2, 3, 4])), f"label values out of range: {uniq}"

    new = {'eeg': eeg, 'label': label, 'PSD100': psd}
    note = f"EEG {eeg_raw.shape}->{eeg.shape}, label {tuple(np.asarray(m['label']).shape)}->{label.shape}, labels={uniq.astype(int).tolist()}"
    return new, note


def main():
    ap = argparse.ArgumentParser(description='Standardize SHHS .mat into the unified eeg/label/PSD100 format')
    ap.add_argument('--apply', action='store_true', help='actually write files (default is dry-run only)')
    args = ap.parse_args()

    mode = 'APPLY (writing files)' if args.apply else 'DRY-RUN (read-only, no writes)'
    device = pick_device()
    print(f"==== Mode: {mode} | device: {device} ====")
    print(f"Input:  {SRC_DIR}")
    print(f"Output: {DST_DIR}")

    if args.apply:
        os.makedirs(DST_DIR, exist_ok=True)

    stft = STFT(n_fft=256, hop_length=100, win_length=100).to(device)

    files = real_mat_files(SRC_DIR)
    print(f"\n=== shhs1: {len(files)} files ===")
    n_files = n_err = 0
    for i, fn in enumerate(files):
        src = os.path.join(SRC_DIR, fn)
        try:
            m = loadmat(src)
            new, note = transform(m, stft, device)
        except Exception as e:
            n_err += 1
            print(f"  !! {fn}: {e}")
            continue
        n_files += 1
        if i < 2:  # print first two samples
            shapes = {k: v.shape for k, v in new.items()}
            print(f"  [sample] {fn}: {note}")
            print(f"           -> keys/shapes {shapes}")
        if args.apply:
            dst = os.path.join(DST_DIR, fn)
            tmp = dst + '.tmp'
            savemat(tmp, new, do_compression=False)
            os.replace(tmp, dst)

    print(f"\n==== Summary: processed {n_files} files, {n_err} errors ====")
    if not args.apply:
        print("No files were written. Review the output above, then re-run with --apply to write.")


if __name__ == '__main__':
    main()
