"""
One-time standardization of .mat files across three dataset subdirectories,
unifying key names and dimensions into a single pipeline-consumable format:

After standardization, each .mat contains exactly 3 keys:
    eeg     (N, 1, 3000)  float32   — kept as-is
    label   (N, 1)        float32   — unified to column vector (DREAMS reshaped from (1,N))
    PSD100  (N, 30, 128)  float32   — EDF kept as-is; DREAMS renamed from 'PSD'

Also removes the redundant PSD200 key from EDF78/EDF20.

Safety policy: first write to a temp file then os.replace for atomic replacement,
avoiding corruption from mid-write interruption.
Default dry-run (check/print only); pass --apply to actually rewrite.
"""
import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat, savemat

BASE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = ['EDF78', 'EDF20', 'DREAMS']

EXPECT_EEG_TAIL = (1, 3000)
EXPECT_PSD_TAIL = (30, 128)


def real_mat_files(folder):
    d = os.path.join(BASE, folder)
    return [f for f in sorted(os.listdir(d))
            if f.endswith('.mat') and not f.startswith('._')]


def transform(m):
    """Build a standardized dict from a loaded mat dict, with validation. Returns (new_dict, note)."""
    eeg = np.asarray(m['eeg'], dtype=np.float32)
    n = eeg.shape[0]

    # PSD: prefer PSD100, otherwise use PSD (DREAMS)
    if 'PSD100' in m:
        psd = np.asarray(m['PSD100'], dtype=np.float32)
        psd_src = 'PSD100'
    elif 'PSD' in m:
        psd = np.asarray(m['PSD'], dtype=np.float32)
        psd_src = 'PSD'
    else:
        raise KeyError("neither PSD100 nor PSD found")

    # Unify label as (N, 1) column vector, preserving original dtype (float32)
    label_in = np.asarray(m['label'])
    label = label_in.reshape(-1, 1).astype(np.float32)

    # ---- Validation ----
    assert eeg.shape[1:] == EXPECT_EEG_TAIL, f"eeg trailing dims unexpected: {eeg.shape}"
    assert psd.shape[1:] == EXPECT_PSD_TAIL, f"PSD trailing dims unexpected: {psd.shape}"
    assert psd.shape[0] == n, f"PSD epoch count {psd.shape[0]} != eeg {n}"
    assert label.shape == (n, 1), f"label shape unexpected: {label.shape}"
    # label domain must be 5-class {0..4}
    uniq = np.unique(label)
    assert np.all(np.isin(uniq, [0, 1, 2, 3, 4])), f"label values out of range: {uniq}"

    new = {'eeg': eeg, 'label': label, 'PSD100': psd}
    dropped = sorted(k for k in m.keys()
                     if not k.startswith('__') and k not in ('eeg', 'label', psd_src))
    note = f"psd_src={psd_src}, label {tuple(label_in.shape)}->{label.shape}, dropped={dropped or 'none'}"
    return new, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='actually rewrite files (default is dry-run only)')
    args = ap.parse_args()

    mode = 'APPLY (rewriting files)' if args.apply else 'DRY-RUN (read-only, no writes)'
    print(f"==== Mode: {mode} ====")

    total_before = total_after = 0
    n_files = n_err = 0
    for folder in FOLDERS:
        d = os.path.join(BASE, folder)
        files = real_mat_files(folder)
        print(f"\n=== {folder}: {len(files)} files ===")
        for i, fn in enumerate(files):
            path = os.path.join(d, fn)
            try:
                m = loadmat(path)
                new, note = transform(m)
            except Exception as e:
                n_err += 1
                print(f"  !! {fn}: {e}")
                continue
            n_files += 1
            if i < 2:  # print first two samples per folder
                shapes = {k: v.shape for k, v in new.items()}
                print(f"  [sample] {fn}: {note}")
                print(f"           -> keys/shapes {shapes}")
            sz_before = os.path.getsize(path)
            total_before += sz_before
            if args.apply:
                tmp = path + '.tmp'
                savemat(tmp, new, do_compression=False)
                # atomic replace
                os.replace(tmp, path)
                total_after += os.path.getsize(path)
        if not args.apply:
            print(f"  (dry-run) would unify to keys=['eeg','label','PSD100'], label→(N,1), drop redundant PSD200/PSD keys")

    print(f"\n==== Summary: processed {n_files} files, {n_err} errors ====")
    print(f"Original total size: {total_before/1e9:.2f} GB")
    if args.apply:
        print(f"Rewritten total size: {total_after/1e9:.2f} GB  (saved {(total_before-total_after)/1e9:.2f} GB)")
    else:
        print("No files were written. Review the output above, then re-run with --apply to execute.")


if __name__ == '__main__':
    main()
