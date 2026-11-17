"""
RERUN_FULL - Step 4: Late fusion (weighted soft voting)
=======================================================
Components:
  Clinical      RERUN_FULL/01_Clinical/01_predictions_best.csv   (Step-1 best)
  DeepLearning  RERUN_FULL/03_DeepLearning/03b_dl_predictions_ens.csv
  Habitat       RERUN_FULL/02_Habitat/std_pop/02i_habitat_predictions_intra_peri_pop.csv
                (population-level clustering: k=3 centres learned on pooled
                 Train voxels, fixed boundaries applied to all patients;
                 intra + peri ring, ICC-screened, LASSO 1-SE, 14 features)
                (standard workflow: intra + peritumoral k=3 sub-regions,
                 ICC-screened, LASSO 1-SE, 10 features)

Fusion weight = 0.5 clinical + 0.3 deep learning + 0.2 habitat (5:3:2 soft
voting on class probabilities).

Outputs -> F:/yu/RERUN_FULL/04_Fusion/
  metrics_all_models.csv, fusion_predictions.csv
  bootstrap_vs_components.csv        paired bootstrap, 2000 resamples
  roc_2x2_compare / calibration_2x2_compare / dca_2x2_compare
  confusion_fusion_2x2
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix,
                             accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)

# shared temperature-scaling implementation (single source of truth; also
# imported by 04b_weight_sensitivity.py and 04c_ablation.py so that all three
# analyses report metrics in the same calibrated probability space)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fusion_common as fc

BASE = 'F:/yu'
ROOT = f'{BASE}/RERUN_FULL'
OUT = f'{ROOT}/04_Fusion'
os.makedirs(OUT, exist_ok=True)

DPI = 300
DS_NAMES = ['Train', 'Internal_Val', 'External_Val_1', 'External_Val_2']
DS_DISPLAY = {'Train': 'Training', 'Internal_Val': 'Internal validation',
              'External_Val_1': 'External validation 1',
              'External_Val_2': 'External validation 2'}
CLASS_NAMES = ['class1', 'class2', 'class3']
WEIGHTS = {'Clinical': 0.5, 'DeepLearning': 0.3, 'Habitat': 0.2}
COLORS = {'Clinical': '#4C72B0', 'DeepLearning': '#55A868',
          'Habitat': '#DD8452', 'Fusion': '#C44E52'}
RNG = np.random.RandomState(42)


def load_norm(path, drop_cols=()):
    d = pd.read_csv(path).drop(columns=list(drop_cols), errors='ignore')
    d = d.rename(columns={'Predicted_class': 'Pred_class'})
    keep = ['ID', 'Dataset', 'True_class', 'Pred_class',
            'p_class1', 'p_class2', 'p_class3']
    return d[keep]


# ------------------------------------------------------------------ load
clin = load_norm(f'{ROOT}/01_Clinical/01_predictions_best.csv')
dl = load_norm(f'{ROOT}/03_DeepLearning/03b_dl_predictions_ens.csv')
hab = pd.read_csv(f'{ROOT}/02_Habitat/std_pop/02i_habitat_predictions_intra_peri_pop.csv')
hab = hab[['ID', 'Dataset', 'True_class', 'Pred_class',
           'p_class1', 'p_class2', 'p_class3']]

comps = {'Clinical': clin, 'DeepLearning': dl, 'Habitat': hab}
for k, v in comps.items():
    v['ID'] = v['ID'].astype(int)
    print(f'{k:14s} n={len(v)}  macroAUC(IntVal)='
          f'{roc_auc_score(v[v.Dataset=="Internal_Val"].True_class-1, v[v.Dataset=="Internal_Val"][["p_class1","p_class2","p_class3"]].values, multi_class="ovr", average="macro"):.3f}')

# ------------------------------------------------------------------ fuse
base = clin[['ID', 'Dataset', 'True_class']].copy()
P = {}
for k, v in comps.items():
    m = v.set_index(['Dataset', 'ID']).reindex(
        list(zip(base.Dataset, base.ID)))
    assert m['p_class1'].notna().all(), f'{k}: missing predictions after align'
    P[k] = m[['p_class1', 'p_class2', 'p_class3']].values

Pf_raw = sum(WEIGHTS[k] * P[k] for k in WEIGHTS)
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
print(f'\nFusion weights: {WEIGHTS}  (sum={sum(WEIGHTS.values()):.1f})')

# ---- Temperature scaling of the fusion probabilities ----------------------
# Identical to LAST/Final_NoOrdinal methodology: a single temperature T is
# grid-searched to minimise RPS on the Training set, then applied to the raw
# fusion probabilities before all downstream metrics/figures.
# The implementation lives in _fusion_common.py so that 04b (weight
# sensitivity) and 04c (ablation) can apply the very same T: every reported
# AUROC / RPS then sits in the same probability space as this deployed model.
_ds_arr = base.Dataset.values
_y_arr = base.True_class.values - 1
T_opt = fc.fit_temperature(Pf_raw, _y_arr, _ds_arr, ref='Train')
Pf = fc.temperature_scale_probs(Pf_raw, T_opt)
fc.save_temperature(T_opt, extra={'source': '04_fusion.py',
                                  'model': '5:3:2 weighted soft voting'})
print(f'Fusion temperature scaling: T={T_opt:.3f} (fitted on Train, '
      f'RPS-minimising)')
print(f'  cached -> {fc.TEMPERATURE_JSON}')

fused = base.copy()
fused['Pred_class'] = Pf.argmax(1) + 1
for i in range(3):
    fused[f'p_class{i+1}'] = Pf[:, i]
fused = fused[['ID', 'Dataset', 'True_class', 'Pred_class',
               'p_class1', 'p_class2', 'p_class3']]
fused.to_csv(f'{OUT}/fusion_predictions.csv', index=False)

MODELS = {**comps, 'Fusion': fused}
for k in MODELS:
    MODELS[k] = MODELS[k].set_index(['Dataset', 'ID']).reindex(
        list(zip(base.Dataset, base.ID))).reset_index(drop=True)
    MODELS[k]['Dataset'] = base.Dataset.values
    MODELS[k]['True_class'] = base.True_class.values


# ------------------------------------------------------------------ metrics
def rps_multi(y, P):
    """Ranked probability score (ordinal report metric, lower is better)."""
    K = P.shape[1]
    yoh = np.zeros_like(P)
    yoh[np.arange(len(y)), y] = 1
    cdf_p = np.cumsum(P, axis=1)
    cdf_y = np.cumsum(yoh, axis=1)
    return np.mean(np.sum((cdf_p - cdf_y) ** 2, axis=1) / (K - 1))


rows = []
for name, d in MODELS.items():
    for ds in DS_NAMES:
        s = d[d.Dataset == ds]
        y = s.True_class.values - 1
        P_ = s[['p_class1', 'p_class2', 'p_class3']].values
        yh = P_.argmax(1)
        aucs = [roc_auc_score((y == c).astype(int), P_[:, c]) for c in range(3)]
        rows.append({'Model': name, 'Dataset': ds, 'N': len(y),
                     'AUC_macro': np.mean(aucs), 'AUC_class1': aucs[0],
                     'AUC_class2': aucs[1], 'AUC_class3': aucs[2],
                     'Accuracy': accuracy_score(y, yh),
                     'Balanced_acc': balanced_accuracy_score(y, yh),
                     'F1_macro': f1_score(y, yh, average='macro',
                                          zero_division=0),
                     'Precision_macro': precision_score(y, yh, average='macro',
                                                        zero_division=0),
                     'Recall_macro': recall_score(y, yh, average='macro',
                                                  zero_division=0),
                     'RPS': rps_multi(y, P_)})
metrics = pd.DataFrame(rows)
metrics.round(4).to_csv(f'{OUT}/metrics_all_models.csv', index=False)
print('\n' + metrics.pivot(index='Model', columns='Dataset',
                            values='AUC_macro').round(4).to_string())
print('\nRPS (lower better):')
print(metrics.pivot(index='Model', columns='Dataset', values='RPS')
      .round(4).to_string())

# ------------------------------------------------------------------ bootstrap
def macro_auc(y, P):
    return np.mean([roc_auc_score((y == c).astype(int), P[:, c])
                    for c in range(3)])


boot_rows = []
NBOOT = 2000
for ds in DS_NAMES:
    d0 = MODELS['Fusion']
    mask = (d0.Dataset == ds).values
    y = d0.True_class.values[mask] - 1
    Pf_ = d0[['p_class1', 'p_class2', 'p_class3']].values[mask]
    n = len(y)
    for other in ['Clinical', 'DeepLearning', 'Habitat']:
        Po = MODELS[other][['p_class1', 'p_class2', 'p_class3']].values[mask]
        da, dr = [], []
        for _ in range(NBOOT):
            idx = RNG.randint(0, n, n)
            yb = y[idx]
            if len(np.unique(yb)) < 3:
                continue
            da.append(macro_auc(yb, Pf_[idx]) - macro_auc(yb, Po[idx]))
            dr.append(rps_multi(yb, Pf_[idx]) - rps_multi(yb, Po[idx]))
        da, dr = np.array(da), np.array(dr)
        boot_rows.append({
            'Dataset': ds, 'Comparison': f'Fusion vs {other}',
            'dAUC_mean': da.mean(), 'dAUC_CI_lo': np.percentile(da, 2.5),
            'dAUC_CI_hi': np.percentile(da, 97.5),
            'p_AUC': 2 * min((da <= 0).mean(), (da >= 0).mean()),
            'dRPS_mean': dr.mean(), 'dRPS_CI_lo': np.percentile(dr, 2.5),
            'dRPS_CI_hi': np.percentile(dr, 97.5),
            'p_RPS': 2 * min((dr <= 0).mean(), (dr >= 0).mean())})
boot = pd.DataFrame(boot_rows)
boot.round(4).to_csv(f'{OUT}/bootstrap_vs_components.csv', index=False)
print('\n' + boot[['Dataset', 'Comparison', 'dAUC_mean', 'p_AUC',
                   'dRPS_mean', 'p_RPS']].round(4).to_string(index=False))

# ------------------------------------------------------------------ figures
def plot_roc_compare():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5), dpi=DPI)
    for ds, ax in zip(DS_NAMES, axes.ravel()):
        d0 = MODELS['Fusion']
        y = d0[d0.Dataset == ds].True_class.values - 1
        for name in ['Clinical', 'DeepLearning', 'Habitat', 'Fusion']:
            P_ = MODELS[name][MODELS[name].Dataset == ds][
                ['p_class1', 'p_class2', 'p_class3']].values
            mf = np.linspace(0, 1, 200)
            tprs = []
            for c in range(3):
                fpr, tpr, _ = roc_curve((y == c).astype(int), P_[:, c])
                tprs.append(np.interp(mf, fpr, tpr))
            mt = np.mean(tprs, 0); mt[-1] = 1.0
            ax.plot(mf, mt, color=COLORS[name],
                    lw=2.6 if name == 'Fusion' else 1.6,
                    label=f'{name} (AUC={macro_auc(y, P_):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
        ax.set_title(DS_DISPLAY[ds], fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle('ROC comparison - clinical / DL / habitat / fusion',
                 fontsize=15, fontweight='bold', y=1.0)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/roc_2x2_compare.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def expected_calibration_error(q, y01, n_bins=10, strategy='quantile'):
    """ECE for a binary probability vector q vs true 0/1 labels y01."""
    q = np.asarray(q, dtype=float)
    y01 = np.asarray(y01, dtype=float)
    if strategy == 'quantile':
        edges = np.quantile(q, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, 0
    for i in range(n_bins):
        m = (q > edges[i]) & (q <= edges[i + 1])
        if m.sum():
            e += m.sum() * abs(q[m].mean() - y01[m].mean())
            n += m.sum()
    return e / n if n else np.nan


def pooled_cal_points(P, y, n_bins=10):
    """Pool 3 one-vs-rest tasks; quantile bins over all n*3 predicted probs."""
    q = P.ravel()
    t = np.column_stack([(y == c).astype(float) for c in range(3)]).ravel()
    edges = np.quantile(q, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    xs, ys_, ns = [], [], []
    for i in range(n_bins):
        m = (q > edges[i]) & (q <= edges[i + 1])
        if m.sum() == 0:
            continue
        xs.append(q[m].mean())
        ys_.append(t[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys_), np.array(ns)


def pooled_ece(P, y):
    """Macro ECE: mean of per-class one-vs-rest ECE (quantile bins)."""
    return float(np.mean([expected_calibration_error(P[:, c],
                                                     (y == c).astype(float))
                          for c in range(3)]))


def plot_calibration_compare():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5), dpi=DPI)
    ece_rows, pt_rows = [], []
    for ds, ax in zip(DS_NAMES, axes.ravel()):
        d0 = MODELS['Fusion']
        y = d0[d0.Dataset == ds].True_class.values - 1
        txt = {}
        for name in ['Clinical', 'DeepLearning', 'Habitat', 'Fusion']:
            P_ = MODELS[name][MODELS[name].Dataset == ds][
                ['p_class1', 'p_class2', 'p_class3']].values
            e = pooled_ece(P_, y)
            txt[name] = f'{e:.3f}'
            ece_rows.append({'Dataset': ds, 'Model': name, 'macro_ECE': round(e, 4)})
            xs, ys_, ns = pooled_cal_points(P_, y)
            for x, yv, n_ in zip(xs, ys_, ns):
                pt_rows.append({'Dataset': ds, 'Model': name,
                                'bin_pred': round(float(x), 4),
                                'bin_obs': round(float(yv), 4), 'bin_n': n_})
            ax.plot(xs, ys_, 'o-', color=COLORS[name], ms=4,
                    lw=2.4 if name == 'Fusion' else 1.3,
                    label=name)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.text(0.02, 0.98, 'macro ECE\n' + '\n'.join(f'{k}: {v}' for k, v in txt.items()),                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.85))
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency')
        ax.set_title(DS_DISPLAY[ds], fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8.5); ax.grid(alpha=0.3)
    fig.suptitle('Calibration curves (macro-averaged over the 3 classes)',
                 fontsize=15, fontweight='bold', y=1.0)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/calibration_2x2_compare.{ext}', dpi=DPI,
                    bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(ece_rows).to_csv(f'{OUT}/ece_2x2_compare.csv', index=False)
    pd.DataFrame(pt_rows).to_csv(f'{OUT}/calibration_2x2_compare_points.csv',
                                 index=False)
    print('\nMacro ECE (quantile bins, mean of 3 one-vs-rest classes):')
    print(pd.DataFrame(ece_rows).pivot(index='Model', columns='Dataset',
                                       values='macro_ECE').round(4).to_string())


def plot_dca_compare():
    thrs = np.linspace(0.01, 0.99, 50)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5), dpi=DPI)
    for ds, ax in zip(DS_NAMES, axes.ravel()):
        d0 = MODELS['Fusion']
        y = d0[d0.Dataset == ds].True_class.values - 1
        for name in ['Clinical', 'DeepLearning', 'Habitat', 'Fusion']:
            P_ = MODELS[name][MODELS[name].Dataset == ds][
                ['p_class1', 'p_class2', 'p_class3']].values
            nbs = []
            for t in thrs:
                nb_c = []
                for c in range(3):
                    y01 = (y == c).astype(int)
                    sb = P_[:, c] >= t
                    tp = np.sum(sb & (y01 == 1)); fp = np.sum(sb & (y01 == 0))
                    nb_c.append(tp / len(y01) - fp / len(y01) * (t / (1 - t)))
                nbs.append(np.mean(nb_c))
            ax.plot(thrs, nbs, color=COLORS[name],
                    lw=2.6 if name == 'Fusion' else 1.4, label=name)
        prev = 1 / 3
        ax.plot(thrs, [prev - (1 - prev) * (t / (1 - t)) for t in thrs],
                color='gray', ls=':', lw=1.2, label='Treat all')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_ylim(-0.2, 0.35); ax.set_xlim(0, 1)
        ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
        ax.set_title(DS_DISPLAY[ds], fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8.5); ax.grid(alpha=0.3)
    fig.suptitle('Decision curve analysis (macro average of 3 classes)',
                 fontsize=15, fontweight='bold', y=1.0)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/dca_2x2_compare.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_fusion():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5), dpi=DPI)
    d0 = MODELS['Fusion']
    for ds, ax in zip(DS_NAMES, axes.ravel()):
        s = d0[d0.Dataset == ds]
        cm = confusion_matrix(s.True_class, s.Pred_class, labels=[1, 2, 3])
        ax.imshow(cm, cmap='Reds')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=13,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{DS_DISPLAY[ds]} (Acc={np.trace(cm)/cm.sum():.3f})',
                     fontsize=13, fontweight='bold')
    fig.suptitle('Fusion model (5:3:2) - confusion matrices',
                 fontsize=15, fontweight='bold', y=1.0)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/confusion_fusion_2x2.{ext}', dpi=DPI,
                    bbox_inches='tight')
    plt.close(fig)


plot_roc_compare()
plot_calibration_compare()
plot_dca_compare()
plot_confusion_fusion()

# ------------------------------------------------------------------ summary
f_iv = metrics[(metrics.Model == 'Fusion') & (metrics.Dataset == 'Internal_Val')].iloc[0]
with open(f'{OUT}/04_fusion_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f'''LATE FUSION (WEIGHTED SOFT VOTING) - RERUN_FULL
==============================================
Clinical 0.5  +  Deep learning 0.3  +  Habitat 0.2

Fusion probabilities temperature-scaled: T={T_opt:.3f}
(grid 0.4-4.0, fitted on Train by minimising RPS; identical to
LAST/Final_NoOrdinal methodology; AUROC/accuracy unaffected)

Components
  Clinical      Step-1 best model (RF) predictions
  DeepLearning  EfficientNet-B0-3D, 3-seed ensemble + 8-flip TTA
  Habitat       habitat radiomics (population-level k=3 clustering,
                fixed HU boundaries -570.3/-207.4 learned on Train only;
                intra + 3-mm ring, ICC-screened, LASSO 1-SE, 14 features)

Macro AUC by model and dataset
''')
    f.write(metrics.pivot(index='Model', columns='Dataset',
                          values='AUC_macro').round(4).to_string() + '\n\n')
    f.write('RPS by model and dataset (lower is better)\n')
    f.write(metrics.pivot(index='Model', columns='Dataset',
                          values='RPS').round(4).to_string() + '\n\n')
    f.write('Paired bootstrap (2000 resamples), fusion vs each component\n')
    f.write(boot.round(4).to_string(index=False) + '\n\n')
    f.write('Full metrics\n')
    f.write(metrics.round(4).to_string(index=False) + '\n')

print('\nDONE ->', OUT)
