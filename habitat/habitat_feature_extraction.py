"""
RERUN_FULL - Step 2h: POPULATION-LEVEL habitat clustering (sensitivity arm)
===========================================================================
Comparison arm to 02a (per-patient clustering).  Here the k=3 cluster
centres are learned ONCE on pooled HU voxels from the TRAINING set only
(no leakage), then applied to every patient with fixed thresholds.

  Phase A  : sample <= 2000 voxels/patient from Train tumours, pool,
             KMeans(k=3, n_init=10, random_state=0) on 1-D HU
             -> sorted centres c1<c2<c3, decision boundaries = midpoints
  Phase B  : for each patient (all 4 datasets), voxels are assigned by
             the fixed boundaries (intra + 3-mm peri ring), each
             sub-region -> PyRadiomics Original 107 features + fraction

Output -> 02_Habitat/features/05_habitat_radiomics_features_pop.csv
          02_Habitat/features/pop_kmeans_centers.npz
Run with the py39 interpreter.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
import SimpleITK as sitk
from scipy import ndimage
from sklearn.cluster import KMeans

os.add_dll_directory(r'C:\ProgramData\anaconda3\envs\py39\Library\bin')
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)
logging.getLogger('radiomics').setLevel(logging.ERROR)
from radiomics import featureextractor

BASE = 'F:/yu'
SPLIT = f'{BASE}/RERUN_FULL/00_Split/00_dataset_assignments.csv'
OUT = f'{BASE}/RERUN_FULL/02_Habitat'
FEAT_DIR = f'{OUT}/features'
CT_DIR, MASK_DIR = f'{BASE}/CT', f'{BASE}/mask'
os.makedirs(FEAT_DIR, exist_ok=True)

K = 3
DILATION_MM = 3.0
PAD = 5
MIN_VOXELS = 10
N_WORKERS = 8
VOX_PER_PATIENT = 2000        # sampling cap for the pooled Train fit
KMEANS_NINIT = 10

FPATH = f'{FEAT_DIR}/05_habitat_radiomics_features_pop.csv'
CENTERS_PATH = f'{FEAT_DIR}/pop_kmeans_centers.npz'

# filled in each worker process via pool initializer (spawn-safe)
BOUNDS = None                 # shape (K-1,) ascending midpoints


def _init_worker():
    """Load fixed population boundaries inside every spawned process."""
    global BOUNDS
    z = np.load(CENTERS_PATH)
    BOUNDS = z['bounds']


def build_extractor():
    ex = featureextractor.RadiomicsFeatureExtractor()
    ex.enableAllFeatures()
    ex.enableImageTypes(Original={})
    ex.settings['binWidth'] = 25
    ex.settings['normalize'] = False
    ex.settings['label'] = 1
    ex.settings['geometryTolerance'] = 1e-3
    ex.settings['minimumROIDimensions'] = 1
    return ex


def extract_region(ct_img, ct_np, region_mask, prefix, ex):
    """Assign region voxels to global habitat bins and radiomics each."""
    out = {}
    n = int(region_mask.sum())
    out[f'{prefix}_n_voxels'] = n
    if n < K:
        return out
    vals = ct_np[region_mask]
    lab = np.digitize(vals, BOUNDS)          # 0..K-1, already coldest->densest
    for hi in range(K):
        sel = (lab == hi)
        nv = int(sel.sum())
        out[f'{prefix}_hab{hi+1}_voxel_fraction'] = nv / n
        if nv < MIN_VOXELS:
            continue
        sub = np.zeros(region_mask.shape, np.uint8)
        sub[region_mask] = sel.astype(np.uint8)
        mi = sitk.GetImageFromArray(sub)
        mi.CopyInformation(ct_img)
        try:
            res = ex.execute(ct_img, mi)
        except Exception:
            continue
        for kk, vv in res.items():
            if kk.startswith('diagnostic'):
                continue
            try:
                out[f'{prefix}_hab{hi+1}_{kk}'] = float(vv)
            except Exception:
                out[f'{prefix}_hab{hi+1}_{kk}'] = np.nan
    return out


def worker(pid):
    try:
        ct_path = os.path.join(CT_DIR, f'{pid}.nii.gz')
        mask_path = os.path.join(MASK_DIR, f'{pid}.nii.gz')
        if not (os.path.exists(ct_path) and os.path.exists(mask_path)):
            return (pid, None)

        reader = sitk.ImageFileReader()
        reader.SetFileName(ct_path)
        reader.ReadImageInformation()
        spacing = reader.GetSpacing()

        m = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        m = (m > 0).astype(np.uint8)
        if m.sum() < 4:
            return (pid, None)

        coords = np.where(m > 0)
        zmin = max(0, coords[0].min() - PAD)
        zmax = min(m.shape[0], coords[0].max() + 1 + PAD)
        ymin = max(0, coords[1].min() - PAD)
        ymax = min(m.shape[1], coords[1].max() + 1 + PAD)
        xmin = max(0, coords[2].min() - PAD)
        xmax = min(m.shape[2], coords[2].max() + 1 + PAD)

        m_crop = m[zmin:zmax, ymin:ymax, xmin:xmax]
        reader.SetExtractSize([int(xmax - xmin), int(ymax - ymin),
                               int(zmax - zmin)])
        reader.SetExtractIndex([int(xmin), int(ymin), int(zmin)])
        ct_img = reader.Execute()
        ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float64)

        sp = np.array(spacing[::-1])
        r_dil = int(np.ceil(DILATION_MM / sp.min()))
        dil = ndimage.binary_dilation(m_crop > 0, iterations=r_dil)
        ring = dil & ~(m_crop > 0)

        ex = build_extractor()
        feats = {}
        feats.update(extract_region(ct_img, ct_np, m_crop > 0, 'intra', ex))
        if int(ring.sum()) >= K:
            feats.update(extract_region(ct_img, ct_np, ring, 'peri', ex))
        else:
            feats['peri_n_voxels'] = int(ring.sum())

        feats['ID'] = pid
        return (pid, feats)
    except Exception:
        return (pid, None)


def sample_voxels(pid):
    """Sample <= VOX_PER_PATIENT tumour HU voxels for one patient."""
    import zlib
    mp_ = os.path.join(MASK_DIR, f'{pid}.nii.gz')
    cp = os.path.join(CT_DIR, f'{pid}.nii.gz')
    if not (os.path.exists(cp) and os.path.exists(mp_)):
        return None
    m = sitk.GetArrayFromImage(sitk.ReadImage(mp_))
    m = (m > 0)
    if m.sum() < 10:
        return None
    coords = np.where(m)
    zmin = max(0, coords[0].min() - 2)
    zmax = min(m.shape[0], coords[0].max() + 3)
    ymin = max(0, coords[1].min() - 2)
    ymax = min(m.shape[1], coords[1].max() + 3)
    xmin = max(0, coords[2].min() - 2)
    xmax = min(m.shape[2], coords[2].max() + 3)
    r = sitk.ImageFileReader()
    r.SetFileName(cp)
    r.SetExtractSize([int(xmax - xmin), int(ymax - ymin), int(zmax - zmin)])
    r.SetExtractIndex([int(xmin), int(ymin), int(zmin)])
    ct = sitk.GetArrayFromImage(r.Execute()).astype(np.float64)
    mc = m[zmin:zmax, ymin:ymax, xmin:xmax]
    hu = ct[mc]
    if len(hu) > VOX_PER_PATIENT:
        rng = np.random.RandomState(zlib.crc32(pid.encode('utf-8')))
        hu = rng.choice(hu, VOX_PER_PATIENT, replace=False)
    return hu


def fit_population_centers(train_pids):
    """Phase A: pool sampled Train voxels and learn global centres."""
    t0 = time.time()
    chunks = []
    with mp.Pool(N_WORKERS) as pool:
        for i, hu in enumerate(pool.imap_unordered(sample_voxels, train_pids,
                                                   chunksize=16)):
            if hu is not None:
                chunks.append(hu)
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f'  sampled {i+1}/{len(train_pids)} train patients '
                      f'({len(chunks)} used) elapsed={el:.0f}s', flush=True)
    vals = np.concatenate(chunks)
    print(f'Pooled Train voxels: {len(vals):,} from {len(chunks)} patients '
          f'(sampling {time.time()-t0:.0f}s)', flush=True)
    km = KMeans(n_clusters=K, n_init=KMEANS_NINIT,
                random_state=0).fit(vals.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.flatten())
    bounds = (centers[:-1] + centers[1:]) / 2.0
    np.savez(CENTERS_PATH, centers=centers, bounds=bounds,
             inerti=float(km.inertia_))
    print('Population centres (HU, cold->dense): '
          + ', '.join(f'{c:.1f}' for c in centers))
    print('Fixed boundaries (HU): ' + ', '.join(f'{b:.1f}' for b in bounds))
    return centers, bounds


if __name__ == '__main__':
    assign = pd.read_csv(SPLIT)
    train_pids = assign.loc[assign.Dataset == 'Train', 'ID'].astype(str).tolist()
    pids = assign['ID'].astype(str).tolist()

    smoke = '--smoke' in sys.argv

    if os.path.exists(CENTERS_PATH) and '--refit' not in sys.argv:
        z = np.load(CENTERS_PATH)
        centers, bounds = z['centers'], z['bounds']
        print('Loaded existing population centres: '
              + ', '.join(f'{c:.1f}' for c in centers))
    else:
        print('Phase A: fitting population-level KMeans on Train voxels')
        centers, bounds = fit_population_centers(train_pids)

    BOUNDS = bounds

    if smoke:
        print(f'SMOKE TEST ({len(pids[:8])} patients)')
        for pid in pids[:8]:
            p, f = worker(pid)
            if f is None:
                print(f'  {pid}: FAILED')
                continue
            fr = [round(f.get(f'intra_hab{i}_voxel_fraction', -1), 3)
                  for i in (1, 2, 3)]
            print(f'  {pid}: cols={len(f)-1} intra fracs={fr}')
        print('SMOKE DONE')
        sys.exit(0)

    print(f'\nPhase B: extracting {len(pids)} patients, {N_WORKERS} workers')
    t0 = time.time()
    results, failed = [], 0
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for i, (pid, feats) in enumerate(pool.imap_unordered(worker, pids)):
            if feats is None:
                failed += 1
            else:
                results.append(feats)
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f'  {i+1}/{len(pids)}  failed={failed}  '
                      f'elapsed={el:.0f}s  ETA={el/(i+1)*(len(pids)-i-1):.0f}s',
                      flush=True)

    hf = pd.DataFrame(results)
    hf.to_csv(FPATH, index=False)
    n_feat = len([c for c in hf.columns if c != 'ID'])
    print(f'\nSaved {hf.shape[0]} patients x {n_feat} features -> {FPATH}')
    print(f'Failed: {failed} | total time {time.time()-t0:.0f}s')

    # empty-subregion audit: how often is a global habitat absent?
    fr_cols = [c for c in hf.columns if c.endswith('voxel_fraction')]
    audit = (hf[fr_cols] == 0).mean().round(4)
    print('\nFraction of patients where a habitat is EMPTY (0 voxels):')
    print(audit.to_string())
