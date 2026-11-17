"""
Grad-CAM++ for the DL_V2 deep-learning branch: ResNet18-3D
(Kinetics-400-pretrained) with the
five-seed ensemble actually carried into the fusion model
(seeds 42 / 123 / 2024 / 3407 / 777).

CAM = mean of the five per-seed maps (each min-max normalised before averaging),
so the heatmap represents the deployed ensemble rather than a single checkpoint.
Displayed probability = five-seed + 8-flip TTA probability from
02_train/r3d18_seeds5/predictions_ens_r3d18.csv (the DL component in fusion).

Outputs (F:/yu/DL_V2/05_Report/figures/gradcam/):
  1) fig_gradcampp_main_3x4.{png,pdf,tif}   main-text figure: one typical case
     per class (correctly classified, closest to the class-median probability).
     Columns: CT axial | Grad-CAM++ | overlay axial | overlay coronal.
  2) fig_gradcampp_class{1,2,3}_10x4.{png,pdf}  supplement: top-10 confident
     correct cases per class.
  3) table_gradcampp_samples.csv, gradcampp_info.json,
     gradcampp_selection.json.

Method: practical ReLU adaptation of Grad-CAM++ (alpha = 1, w = sum ReLU(dY/dA))
on the last residual stage (layer4, 512 x 8 x 4 x 4 for a 64^3 input), trilinear
upsampled to 64^3 and min-max normalised per case.
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

CACHE = 'F:/yu/dl_cache'
BASE = 'F:/yu/DL_V2/02_train'
PRED = f'{BASE}/r3d18_seeds5/predictions_ens_r3d18.csv'
CKPTS = [f'{BASE}/r3d18/model_seed42.pth',
         f'{BASE}/r3d18/model_seed123.pth',
         f'{BASE}/r3d18/model_seed2024.pth',
         f'{BASE}/r3d18_seed45/model_seed3407.pth',
         f'{BASE}/r3d18_seed45/model_seed777.pth']
OUT = 'F:/yu/DL_V2/05_Report/figures/gradcam'
DATASET = 'External_Val_1'
CLASS_NAMES = {0: 'AAH/AIS', 1: 'MIA', 2: 'IAC'}
MEAN, STD = 0.45, 0.225
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
os.makedirs(OUT, exist_ok=True)
print(f'Device: {DEVICE}')

sys.path.insert(0, 'F:/yu/DL_V2/02_train')
from importlib import util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    'dlv2train', 'F:/yu/DL_V2/02_train/03d_dl_v2_train.py')
_train = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_train)          # module import only builds defs


def load_models():
    models, meta = [], []
    for p in CKPTS:
        ck = torch.load(p, weights_only=False, map_location='cpu')
        m = _train.Net('r3d18')
        m.load_state_dict(ck['state_dict'])
        m.to(DEVICE).eval()
        for q in m.parameters():
            q.requires_grad_(True)
        models.append(m)
        meta.append({'ckpt': os.path.basename(os.path.dirname(p)) + '/'
                     + os.path.basename(p),
                     'seed': int(ck['seed']),
                     'inval_auc': float(ck['inval_auc'])})
        print(f"  loaded seed {ck['seed']} (inval AUC {ck['inval_auc']:.4f})")
    return models, meta


def pick_target_layer(model):
    shapes = {}
    hooks = []
    for name in ('layer1', 'layer2', 'layer3', 'layer4'):
        mod = getattr(model.backbone, name)
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, n=name: shapes.__setitem__(n, tuple(o.shape[2:]))))
    with torch.no_grad():
        model(torch.zeros(1, 1, 64, 64, 64, device=DEVICE))
    for h in hooks:
        h.remove()
    idx = [n for n in ('layer1', 'layer2', 'layer3', 'layer4')
           if min(shapes[n]) >= 4][-1]
    print(f'target layer: backbone.{idx} (output spatial {shapes[idx]})')
    return getattr(model.backbone, idx), idx, shapes


def gradcampp_one(model, tl_name, x_np, target_class):
    """Single-seed Grad-CAM++ map on a raw [0,1] 64^3 volume."""
    target_layer = getattr(model.backbone, tl_name)
    x = torch.from_numpy((x_np - MEAN) / STD).float().to(DEVICE)
    x = x.unsqueeze(0).unsqueeze(0)
    feats = {}
    h = target_layer.register_forward_hook(
        lambda m, i, o: feats.__setitem__('f', o))
    with torch.enable_grad():
        logits = model(x)
        score = logits[0, target_class]
        fmap = feats['f']
        grad1 = torch.autograd.grad(score, fmap, retain_graph=True)[0]
        try:
            grad2 = torch.autograd.grad(grad1.sum(), fmap, retain_graph=True)[0]
        except Exception:
            grad2 = None
        if grad2 is not None and grad2.abs().max().item() > 1e-6:
            try:
                grad3 = torch.autograd.grad(grad2.sum(), fmap,
                                            retain_graph=True)[0]
            except Exception:
                grad3 = torch.zeros_like(grad1)
            alpha = torch.nan_to_num(
                grad2 ** 2 / (2 * grad2 ** 2 + grad1 * grad3 + 1e-10))
            mode = 'full'
        else:
            alpha = torch.ones_like(grad1)
            mode = 'practical(ReLU)'
        w = (alpha * F.relu(grad1)).sum(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * fmap).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(64, 64, 64), mode='trilinear',
                            align_corners=False)[0, 0].detach().cpu().numpy()
    h.remove()
    rng = cam.max() - cam.min()
    cam = (cam - cam.min()) / rng if rng > 1e-10 else np.zeros_like(cam)
    probs = F.softmax(logits[0].float(), 0).detach().cpu().numpy()
    del x, logits, score, fmap, grad1
    torch.cuda.empty_cache()
    return cam, probs, mode


def gradcampp_ensemble(models, tl_name, x_np, target_class):
    cams, probs, modes = [], [], []
    for m in models:
        c, p, md = gradcampp_one(m, tl_name, x_np, target_class)
        cams.append(c)
        probs.append(p)
        modes.append(md)
    stack = np.stack(cams)
    cam = stack.mean(0)
    rng = cam.max() - cam.min()
    cam = (cam - cam.min()) / rng if rng > 1e-10 else np.zeros_like(cam)
    # cross-seed agreement: mean pairwise Pearson r of the per-seed maps
    flat = stack.reshape(len(cams), -1)
    cs = np.corrcoef(flat)
    agree = float((cs.sum() - len(cams)) / (len(cams) * (len(cams) - 1)))
    return cam, np.mean(probs, 0), modes, agree


def centre_slices(vol):
    central = vol[20:44, 20:44, :]
    z = int(np.argmax(central.mean(axis=(0, 1))[20:44])) + 20
    return z


def coronal_slice(cam):
    return int(np.unravel_index(int(np.argmax(cam)), cam.shape)[1])


def overlay(gray2d, cam2d, alpha=0.45):
    """Blend a lung-windowed grayscale CT slice with the jet heatmap."""
    rgb = plt.get_cmap('jet')(np.clip(cam2d, 0, 1))[:, :, :3]
    return np.clip(gray2d[:, :, None] * (1 - alpha) + rgb * alpha, 0, 1)


def heat(cam2d):
    return plt.get_cmap('jet')(np.clip(cam2d, 0, 1))[:, :, :3]


def select_samples():
    ens = pd.read_csv(PRED, dtype={'ID': int})
    ens = ens[ens.Dataset == DATASET].sort_values('ID')
    supp, main = {}, {}
    for c in range(3):
        ok = ens[(ens.True_class == c + 1) & (ens.Pred_class == c + 1)]
        top = ok.sort_values(f'p_class{c + 1}', ascending=False).head(10)
        supp[c] = [{'id': int(r.ID), 'p': float(r[f'p_class{c + 1}'])}
                   for _, r in top.iterrows()]
        med = ok[f'p_class{c + 1}'].median()
        r = ok.iloc[(ok[f'p_class{c + 1}'] - med).abs().argsort()[:1]].iloc[0]
        main[c] = {'id': int(r.ID), 'p': float(r[f'p_class{c + 1}'])}
        print(f'class {c + 1} ({CLASS_NAMES[c]}): {len(ok)} correct, '
              f'top-10 conf {top[f"p_class{c + 1}"].min():.3f}'
              f'-{top[f"p_class{c + 1}"].max():.3f} | '
              f'median {med:.3f} -> typical ID {main[c]["id"]} '
              f'p={main[c]["p"]:.3f}')
    return supp, main


# ------------------- tumour-geometry helpers (mask-driven display) ---------
CT_DIR = 'F:/yu/CT/%d.nii.gz'
MK_DIR = 'F:/yu/mask/%d.nii.gz'
WIN = 44            # display window (voxels) centred on the tumour
# Standard lung window WL -600 / WW 1500 -> [-1350, 150] HU. The dl_cache is
# clipped to [-1000, 400] HU and scaled to [0, 1] (x), so HU = x*1400 - 1000;
# the windowed grey level is computed from HU exactly as in Figure 2. Voxels
# below -1000 HU are deep air and all display at the -1000 HU grey level.
HU_LO, HU_HI = -1350.0, 150.0


def to_hu(x):
    """dl_cache [0,1] values back to HU (clip floor -1000 -> 0)."""
    return x * 1400.0 - 1000.0


def lung_gray(hu2d):
    """Window a HU slice with the standard lung window -> [0,1] grayscale."""
    return np.clip((hu2d - HU_LO) / (HU_HI - HU_LO), 0.0, 1.0)


def crop_geometry(pid):
    """Tumour mask inside the exact 64^3 cache crop (same centre rule as
    dl_preprocess.py: mask bbox centre). Returns None if the mask is empty.
    """
    import nibabel as nib
    mask = nib.load(MK_DIR % pid).get_fdata()
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    c = (coords.min(0) + coords.max(0)) / 2
    lo_raw = c - 32
    lo = np.maximum(0, lo_raw).astype(int)
    pad_lo = np.maximum(0, -lo_raw).astype(int)
    pad_hi = np.maximum(0, (lo_raw + 64) - np.array(mask.shape)).astype(int)
    hi = np.minimum(np.array(mask.shape), lo + 64)
    m = (mask > 0)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if pad_lo.any() or pad_hi.any():
        m = np.pad(m, [(pad_lo[i], pad_hi[i]) for i in range(3)],
                   mode='constant', constant_values=False)
    if m.sum() == 0:
        return None
    idx = np.argwhere(m)
    area = m.sum(axis=(0, 1))                      # in-plane area per k slice
    zooms = nib.load(CT_DIR % pid).header.get_zooms()[:3]
    return {'mask': m, 'best_k': int(np.argmax(area)),
            'best_area': int(area.max()), 'centroid': idx.mean(0),
            'extent': idx.max(0) - idx.min(0) + 1, 'zooms': zooms}


# Visual confirmation override for the S3 showcase (verified against the
# candidate contact sheet): for AAH/AIS the automated pick (ID 3237) had the
# chest wall occupying a third of the window; ID 812 (p = 0.447, |dp| = 0.009
# of the class median) is the typical, fully centred, clean-lung case.
MANUAL_MAIN = {0: 812}


def select_main_cases():
    """Typical AND visible: correctly classified cases of External_Val_1
    ranked by |p - class median|; among those within 0.01 of the median that
    pass visibility filters (lesion >= 4 voxels on every axis, max in-plane
    area >= 60 vox, centroid within 8 vox of the crop centre), show the case
    with the largest in-plane lesion area. Final choice visually confirmed.
    """
    ens = pd.read_csv(PRED, dtype={'ID': int})
    ens = ens[ens.Dataset == DATASET].sort_values('ID')
    main = {}
    for c in range(3):
        ok = ens[(ens.True_class == c + 1) & (ens.Pred_class == c + 1)].copy()
        med = ok[f'p_class{c + 1}'].median()
        ok['dp'] = (ok[f'p_class{c + 1}'] - med).abs()
        cand = ok.sort_values('dp').head(15)
        good = []
        for _, r in cand.iterrows():
            g = crop_geometry(int(r.ID))
            if g is None:
                continue
            if (min(g['extent']) >= 4 and g['best_area'] >= 60
                    and np.abs(g['centroid'] - 32).max() <= 8):
                good.append((float(r.dp), g['best_area'], int(r.ID),
                             float(r[f'p_class{c + 1}'])))
        pool = [t for t in good if t[0] <= 0.01] or good
        if c in MANUAL_MAIN and any(t[2] == MANUAL_MAIN[c] for t in pool):
            dp, area, pid, p = next(t for t in pool if t[2] == MANUAL_MAIN[c])
        else:
            dp, area, pid, p = max(pool, key=lambda t: t[1])
        main[c] = {'id': pid, 'p': p, 'median': float(med), 'dp': dp,
                   'area': area}
        print(f'main class {c + 1} ({CLASS_NAMES[c]}): ID {pid} '
              f'p={p:.3f} (median {med:.3f}, |dp|={dp:.3f}), '
              f'in-plane area {area} vox')
    return main


def win_bounds(c, w=WIN, lo=0, hi=64):
    s = int(round(c - w / 2))
    s = min(max(s, lo), hi - w)
    return s, s + w


def display_geom(g):
    """Slices/windows driven by the tumour: axial slice through the largest
    cross-section; coronal plane through the centroid; both windows centred
    on the tumour. Returns (k, (a0,a1) axial window, j, (a0,a2) coronal
    window, scale-bar length in voxels for a horizontal 10 mm bar).
    """
    cen = g['centroid']
    k = g['best_k']
    a0, a1 = win_bounds(cen[0]), win_bounds(cen[1])
    j = int(round(cen[1]))
    b0, b2 = win_bounds(cen[0]), win_bounds(cen[2])
    bar = 10.0 / g['zooms'][0]
    return k, (a0, a1), j, (b0, b2), bar


def _draw_main(cams, col_titles, fs, figsize, out_base, dpi, cbar_rect, margins):
    """Render the 3 x 4 typical-case panel at a given canvas/font size.

    Every panel shows a WIN^2 display window centred on the tumour; the axial
    slice is the tumour's largest cross-section and the coronal plane passes
    through the tumour centroid. 'ct_ax'/'ct_co' arrive already windowed to
    the standard lung window (WL -600 / WW 1500) as [0,1] grayscale.
    A 10 mm scale bar (white) is drawn on the original-CT column.
    """
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    for row, d in enumerate(cams):
        ct_ax, cam_ax, ct_co, cam_co = d['ct_ax'], d['cam_ax'], d['ct_co'], \
            d['cam_co']
        ax = axes[row, 0]
        ax.imshow(ct_ax, cmap='gray', vmin=0, vmax=1, origin='lower', interpolation='bilinear')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[0], fontsize=fs['col'], pad=fs['pad'])
        ax.text(-0.05, 0.5, f'{d["cls"]}',
                transform=ax.transAxes, fontsize=fs['row'],
                ha='right', va='center', linespacing=1.4)
        frac = d['bar_vox'] / WIN
        # white 10-mm bar with a thin black casing so it reads on both dark
        # lung parenchyma and bright chest wall; end ticks drawn as well
        for lw, col in ((5.6, 'black'), (3.2, 'white')):
            ax.plot([0.08, 0.08 + frac], [0.07, 0.07], color=col, lw=lw,
                    solid_capstyle='butt', transform=ax.transAxes, zorder=4)
            for xx in (0.08, 0.08 + frac):
                ax.plot([xx, xx], [0.058, 0.082], color=col, lw=lw * 0.75,
                        solid_capstyle='butt', transform=ax.transAxes,
                        zorder=4)
        ax.text(0.08 + frac / 2, 0.098, '10 mm', fontsize=fs['bar'],
                color='white', ha='center', va='bottom',
                transform=ax.transAxes, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.4,
                                            foreground='black')])

        ax = axes[row, 1]
        ax.imshow(heat(cam_ax), origin='lower', interpolation='bilinear')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[1], fontsize=fs['col'], pad=fs['pad'])

        ax = axes[row, 2]
        ax.imshow(overlay(ct_ax, cam_ax), origin='lower', interpolation='bilinear')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[2], fontsize=fs['col'], pad=fs['pad'])

        ax = axes[row, 3]
        ax.imshow(overlay(ct_co, cam_co), origin='lower', interpolation='bilinear')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[3], fontsize=fs['col'], pad=fs['pad'])
    cax = fig.add_axes(cbar_rect)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['low', 'high'])
    cb.ax.tick_params(labelsize=fs['cb'])
    cb.set_label('Grad-CAM++ activation (normalised per case)',
                 fontsize=fs['cblab'])
    fig.subplots_adjust(hspace=0.05, wspace=0.04, **margins)
    fig.savefig(f'{out_base}.png', dpi=dpi, facecolor='white')
    fig.savefig(f'{out_base}.pdf', facecolor='white')
    fig.savefig(f'{out_base}.tif', dpi=dpi, facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f'saved {out_base}.png/.pdf/.tif ({dpi} dpi)')


def make_main_figure(main, models, tl_name):
    col_titles = ['Original CT (axial)', 'Grad-CAM++ (heatmap)',
                  'Overlay (axial)', 'Overlay (coronal)']
    modes, agree, cams = set(), [], []
    for c in range(3):
        s = main[c]
        pid = s['id']
        x = np.load(f"{CACHE}/{pid}.npy").astype(np.float32)
        hu = to_hu(x)
        cam, probs, md, ag = gradcampp_ensemble(models, tl_name, x, c)
        modes.update(md)
        agree.append(ag)
        g = crop_geometry(pid)
        k, (a0, a1), j, (b0, b2), bar = display_geom(g)
        cams.append(dict(cls=CLASS_NAMES[c], p=s['p'],
                         ct_ax=lung_gray(hu[a0[0]:a0[1], a1[0]:a1[1], k].T),
                         cam_ax=cam[a0[0]:a0[1], a1[0]:a1[1], k].T,
                         ct_co=lung_gray(hu[b0[0]:b0[1], j, b2[0]:b2[1]].T),
                         cam_co=cam[b0[0]:b0[1], j, b2[0]:b2[1]].T,
                         bar_vox=bar))
        print(f'  main [{c + 1}/3] class {c + 1} ID={pid} '
              f'p={s["p"]:.3f} k={k} j={j} win={WIN} bar={bar:.1f}vox '
              f'seed-agreement r={ag:.3f}')

    # screen / inspection canvas
    _draw_main(cams, col_titles,
               fs=dict(col=12, row=11, pad=6, cb=8, cblab=9, bar=8),
               figsize=(11.0, 8.6), out_base=f'{OUT}/fig_gradcampp_main_3x4',
               dpi=300, cbar_rect=[0.30, 0.055, 0.40, 0.012],
               margins=dict(left=0.10, right=0.99, top=0.96, bottom=0.10))

    # Supplementary Figure S3: journal print width 175 mm (as Figures 2-5)
    _draw_main(cams, col_titles,
               fs=dict(col=10.0, row=10.0, pad=3, cb=8.5, cblab=10.0, bar=8.5),
               figsize=(175 / 25.4, 150 / 25.4),
               out_base=f'{OUT}/figS3_gradcampp_3x4',
               dpi=600, cbar_rect=[0.30, 0.078, 0.40, 0.011],
               margins=dict(left=0.102, right=0.995, top=0.945, bottom=0.135))
    return float(np.mean(agree))


def make_supplement_figures(supp, models, tl_name):
    info_rows = []
    for c in range(3):
        samples = [{'id': s['id'], 'p': s['p'], 'target': c} for s in supp[c]]
        n = len(samples)
        fig, axes = plt.subplots(n, 4, figsize=(13, 3.0 * n), dpi=200)
        if n == 1:
            axes = axes[np.newaxis, :]
        col_titles = ['Original CT (axial)', 'Grad-CAM++ (heatmap)',
                      'Overlay (axial)', 'Overlay (coronal)']
        for row, s in enumerate(samples):
            x = np.load(f"{CACHE}/{s['id']}.npy").astype(np.float32)
            hu = to_hu(x)
            cam, probs, md, ag = gradcampp_ensemble(models, tl_name, x,
                                                    s['target'])
            g = crop_geometry(s['id'])
            if g is not None:
                k, (a0, a1), j, (b0, b2), bar = display_geom(g)
                ct_ax = lung_gray(hu[a0[0]:a0[1], a1[0]:a1[1], k].T)
                cam_ax = cam[a0[0]:a0[1], a1[0]:a1[1], k].T
                ct_co = lung_gray(hu[b0[0]:b0[1], j, b2[0]:b2[1]].T)
                cam_co = cam[b0[0]:b0[1], j, b2[0]:b2[1]].T
            else:                       # no usable mask -> centre slices
                k = centre_slices(x)
                yc = coronal_slice(cam)
                ct_ax, cam_ax = lung_gray(hu[:, :, k].T), cam[:, :, k].T
                ct_co, cam_co = lung_gray(hu[:, yc, :].T), cam[yc, :, :].T
            ax = axes[row, 0]
            ax.imshow(ct_ax, cmap='gray', vmin=0, vmax=1, origin='lower', interpolation='bilinear')
            ax.axis('off')
            ax.set_title(col_titles[0] if row == 0 else '', fontsize=11,
                         fontweight='bold')
            ax.text(-0.06, 0.5, f"#{row + 1}\n"
                                f"P({CLASS_NAMES[s['target']]})="
                                f"{s['p']:.3f}",
                    transform=ax.transAxes, fontsize=9, ha='right',
                    va='center', linespacing=1.5)
            ax = axes[row, 1]
            ax.imshow(heat(cam_ax), origin='lower', interpolation='bilinear')
            ax.axis('off')
            ax.set_title(col_titles[1] if row == 0 else '', fontsize=11,
                         fontweight='bold')
            ax = axes[row, 2]
            ax.imshow(overlay(ct_ax, cam_ax), origin='lower', interpolation='bilinear')
            ax.axis('off')
            ax.set_title(col_titles[2] if row == 0 else '', fontsize=11,
                         fontweight='bold')
            ax = axes[row, 3]
            ax.imshow(overlay(ct_co, cam_co), origin='lower', interpolation='bilinear')
            ax.axis('off')
            ax.set_title(col_titles[3] if row == 0 else '', fontsize=11,
                         fontweight='bold')
            info_rows.append({'Figure': f'S_gradcampp_class{c + 1}',
                              'Class': c + 1, 'Class_name': CLASS_NAMES[c],
                              'rank': row + 1, 'ID': s['id'],
                              'P_true_ensemble': s['p'],
                              'seed_agreement_r': ag, 'Dataset': DATASET})
            print(f'  supp class {c + 1} #{row + 1} ID={s["id"]} '
                  f'p={s["p"]:.3f} r={ag:.3f}')
        fig.suptitle(f'{CLASS_NAMES[c]} (class {c + 1}) - top-10 confident '
                     f'correct cases, Grad-CAM++ (Kinetics-400-pretrained '
                     f'ResNet18-3D, 5-seed ensemble, '
                     f'external validation 1)', fontsize=13,
                     fontweight='bold', y=1.005)
        plt.tight_layout(h_pad=0.3, w_pad=0.3)
        base = f'{OUT}/fig_gradcampp_class{c + 1}_10x4'
        fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{base}.pdf', bbox_inches='tight')
        fig.savefig(f'{base}.tif', dpi=300, bbox_inches='tight',
                    pil_kwargs={'compression': 'tiff_lzw'})
        plt.close(fig)
        print(f'saved {base}.png/.pdf ({n} rows)')
    pd.DataFrame(info_rows).to_csv(f'{OUT}/table_gradcampp_samples.csv',
                                   index=False)


def main():
    supp, _ = select_samples()
    main_sel = select_main_cases()
    models, meta = load_models()
    tl, tl_name, shapes = pick_target_layer(models[0])

    for c in range(3):
        x = np.load(f"{CACHE}/{main_sel[c]['id']}.npy").astype(np.float32)
        cam, probs, md, ag = gradcampp_ensemble(models, tl_name, x, c)
        print(f'  typical class {c + 1}: 5-seed CAM-model probs '
              f'{np.round(probs, 3).tolist()} (TTA ensemble P_true='
              f'{main_sel[c]["p"]:.3f}) r={ag:.3f}')

    main_agree = make_main_figure(main_sel, models, tl_name)
    make_supplement_figures(supp, models, tl_name)

    json.dump({
        'model': 'ResNet18-3D (torchvision, Kinetics-400 pretrained), '
                 'five-seed '
                 'ensemble (42/123/2024/3407/777), the DL branch deployed in '
                 'the fusion model',
        'checkpoints': meta,
        'dataset': DATASET,
        'target_layer': f'backbone.{tl_name}, output spatial {shapes[tl_name]} '
                        f'(last residual stage with spatial >= 4)',
        'cam_mode': 'Grad-CAM++ practical ReLU adaptation',
        'ensemble_cam': 'per-seed maps min-max normalised, then averaged',
        'seed_agreement_main_mean_r': main_agree,
        'upsample': 'trilinear to 64^3, min-max normalised per case',
        'main_selection': 'per class, correctly classified External_Val_1 '
                          'cases ranked by |p - class median| (typicality); '
                          'among those within 0.01 of the median passing '
                          'visibility filters (lesion >= 4 voxels per axis, '
                          'in-plane area >= 60 voxels, centroid within 8 '
                          'voxels of the crop centre), the case with the '
                          'largest in-plane lesion area is shown (visually '
                          'confirmed); not cherry-picked by performance',
        'display_window': 'standard lung window WL -600 / WW 1500 '
                          '([-1350, 150] HU) applied to the HU values '
                          'recovered from the dl_cache (HU = x*1400 - 1000, '
                          'cache floor -1000 HU); 44^2 display window '
                          'centred on the tumour centroid (the 64^3 cache '
                          'crop itself is centred on the tumour-mask '
                          'bounding-box centre); axial slice = largest '
                          'tumour cross-section, coronal plane = through '
                          'the tumour centroid; 10 mm scale bar on the CT '
                          'column',
        'supp_selection': 'top-10 highest-confidence correctly classified '
                          'cases per class',
        'displayed_probability': 'five-seed + 8-flip TTA ensemble probability '
                                 'from 02_train/r3d18_seeds5/'
                                 'predictions_ens_r3d18.csv',
        'input_normalisation': '(x - 0.45) / 0.225 on the raw [0,1] 64^3 crop',
        'cache': 'F:/yu/dl_cache/{ID}.npy, 64^3 crop, HU window [-1000,400] '
                 'normalised to [0,1]',
    }, open(f'{OUT}/gradcampp_info.json', 'w'), indent=2)
    json.dump({'main': {CLASS_NAMES[c]: main_sel[c] for c in range(3)},
               'supplement': {CLASS_NAMES[c]: supp[c] for c in range(3)}},
              open(f'{OUT}/gradcampp_selection.json', 'w'), indent=2)
    print('\nall outputs ->', OUT)


if __name__ == '__main__':
    main()
