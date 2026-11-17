#!/usr/bin/env python3
"""
=============================================================================
深度学习预处理: 从 NIfTI CT+mask 提取以肿瘤为中心的 64x64x64 crop
- HU 裁剪到 [-1000, 400] (肺窗), 归一化到 [0, 1]
- 输出: F:/yu/dl_cache/{ID}.npy (float32, 1x64x64x64) + labels.csv
=============================================================================
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool
warnings.filterwarnings('ignore')

CT_DIR = r'F:\yu\CT'
MASK_DIR = r'F:\yu\mask'
ORIG_DIR = r'F:\yu\original_data'
CACHE_DIR = r'F:\yu\dl_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

CROP_SIZE = 64
HU_MIN, HU_MAX = -1000.0, 400.0
DATASETS = ['Train', 'Internal_Val', 'External_Val_1', 'External_Val_2']
N_WORKERS = 14

def crop_tumor(ct_path, mask_path, size=CROP_SIZE):
    """Crop size^3 box centered on tumor bbox center."""
    ct = nib.load(ct_path).get_fdata().astype(np.float32)
    mask = nib.load(mask_path).get_fdata()
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    center = ((coords.min(axis=0) + coords.max(axis=0)) / 2).astype(int)
    half = size // 2
    # Crop bounds with padding
    lo = center - half
    hi = center + half
    pad_lo = np.maximum(0, -lo)
    pad_hi = np.maximum(0, hi - np.array(ct.shape))
    lo = np.maximum(0, lo)
    hi = np.minimum(ct.shape, hi)
    crop = ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    # Pad with HU_MIN (air)
    if pad_lo.any() or pad_hi.any():
        crop = np.pad(crop, [(pad_lo[i], pad_hi[i]) for i in range(3)],
                      mode='constant', constant_values=HU_MIN)
    # HU window + normalize
    crop = np.clip(crop, HU_MIN, HU_MAX)
    crop = (crop - HU_MIN) / (HU_MAX - HU_MIN)
    return crop.astype(np.float32)

def process_one(args):
    """Worker: process a single case. Returns (ID, Dataset, Output) or None."""
    pid, ds, y = args
    out_path = os.path.join(CACHE_DIR, f'{pid}.npy')
    if os.path.exists(out_path):
        return (pid, ds, y)
    ct_path = os.path.join(CT_DIR, f'{pid}.nii.gz')
    mask_path = os.path.join(MASK_DIR, f'{pid}.nii.gz')
    if not (os.path.exists(ct_path) and os.path.exists(mask_path)):
        return None
    try:
        crop = crop_tumor(ct_path, mask_path)
        if crop is None or crop.shape != (CROP_SIZE,) * 3:
            return None
        np.save(out_path, crop)
        return (pid, ds, y)
    except Exception:
        return None

def main():
    t0 = time.time()
    # Build task list
    tasks = []
    for ds in DATASETS:
        df = pd.read_csv(os.path.join(ORIG_DIR, f'original_data_{ds}.csv'))
        for pid, y in zip(df['ID'].values, df['Output'].values):
            tasks.append((pid, ds, int(y)))
    print(f"Total tasks: {len(tasks)}, workers: {N_WORKERS}")

    rows, n_fail = [], 0
    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, tasks, chunksize=8)):
            if res is None:
                n_fail += 1
            else:
                rows.append({'ID': res[0], 'Dataset': res[1], 'Output': res[2]})
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} done, fail={n_fail}, "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    labels = pd.DataFrame(rows).drop_duplicates('ID')
    labels.to_csv(os.path.join(CACHE_DIR, 'labels.csv'), index=False)
    print(f"\nTotal: {len(rows)} ok, {n_fail} failed, {time.time()-t0:.0f}s")
    print(f"Labels saved: {len(labels)} unique IDs")
    print(labels.groupby('Dataset')['Output'].value_counts().unstack(fill_value=0))

if __name__ == '__main__':
    main()
