"""
RERUN_FULL - Step 2i: Population-level habitat modelling + paradigm comparison
==============================================================================
Same standard funnel as 02g, applied to the population-clustered features
(02h): ICC screening using precomputed ICC estimates (ICC >= 0.75).
Train-only median imputation / variance filter / z-score ->
Spearman |rho|<0.9 -> multinomial LASSO 10-fold 1-SE (2-SE if more than MAX_SEL kept)
-> single multinomial LR.

Arms  : intra_pop (intratumoral only)  and  intra_peri_pop
Also  : paired bootstrap (2000x) of macro AUROC pop-vs-per-patient on all
        datasets, using the 02g prediction files for the per-patient arm.

Input : 02_Habitat/features/05_habitat_radiomics_features_pop.csv
        02_Habitat/features/pop_kmeans_centers.npz
        02_Habitat/icc/icc_stability.csv
        02_Habitat/std/02g_habitat_predictions_{std,extended}.csv
Output: 02_Habitat/std_pop/
"""
import os, warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score, f1_score, r2_score)

warnings.filterwarnings('ignore')

BASE = 'F:/yu'
ROOT = f'{BASE}/RERUN_FULL'
HAB = f'{ROOT}/02_Habitat'
OUT = 'F:/yu/DL_V2/manuscript/RERUN_FULL/02_Habitat/std_pop'
os.makedirs(OUT, exist_ok=True)

DPI = 300
DS_NAMES = ['Train', 'Internal_Val', 'External_Val_1', 'External_Val_2']
DS_DISPLAY = {'Train': 'Training', 'Internal_Val': 'Internal validation',
              'External_Val_1': 'External validation 1',
              'External_Val_2': 'External validation 2'}
CLASS_NAMES = ['class1', 'class2', 'class3']
ICC_THRESH = 0.75
MAX_SEL = 14  # 14-feature signature is the deployed model (no top-10 truncation)
N_BOOT = 2000
RNG = np.random.RandomState(42)


def rps_multi(y, P):
    K = P.shape[1]
    yoh = np.zeros_like(P)
    yoh[np.arange(len(y)), y] = 1
    return np.mean(np.sum((np.cumsum(P, 1) - np.cumsum(yoh, 1)) ** 2, 1) / (K - 1))


def kw_statistics(X, y):
    return np.array([sps.kruskal(*[X[y == c, j] for c in np.unique(y)]).statistic
                     for j in range(X.shape[1])])


def reduce_redundant(X, cols, y, tr_mask, thresh=0.90, tag=''):
    Xtr, ytr = X[tr_mask], y[tr_mask]
    kw = kw_statistics(Xtr, ytr)
    rho = pd.DataFrame(Xtr).corr(method='spearman').abs().values
    kept = []
    for i in np.argsort(-kw):
        if kept and np.max(rho[i, kept]) >= thresh:
            continue
        kept.append(int(i))
    kept = sorted(kept)
    print(f'  [{tag}] collinearity |rho|<{thresh}: {len(cols)} -> {len(kept)}')
    return X[:, kept], [cols[i] for i in kept]


def lasso_select(X, y, cols, tag, se_mult=1.0):
    cv10 = StratifiedKFold(10, shuffle=True, random_state=42)
    Cs = np.logspace(-4, 0, 15)
    fold = {C: [] for C in Cs}
    for C in Cs:
        for tr_i, va_i in cv10.split(X, y):
            lr = LogisticRegression(penalty='l1', C=C, solver='saga',
                                    max_iter=3000, tol=0.01, random_state=42,
                                    class_weight='balanced')
            lr.fit(X[tr_i], y[tr_i])
            if np.all(np.abs(lr.coef_) < 1e-8):
                fold[C].append(0.5)
            else:
                fold[C].append(roc_auc_score(y[va_i], lr.predict_proba(X[va_i]),
                                             multi_class='ovr', average='macro'))
    mean_auc = np.array([np.mean(fold[C]) for C in Cs])
    se = np.array([np.std(fold[C], ddof=1) / np.sqrt(10) for C in Cs])
    best_i = int(np.argmax(mean_auc))
    thr = mean_auc[best_i] - se_mult * se[best_i]
    sel_i = min(i for i in range(len(Cs)) if mean_auc[i] >= thr)

    lr = LogisticRegression(penalty='l1', C=Cs[sel_i], solver='saga',
                            max_iter=4000, tol=0.01, random_state=42,
                            class_weight='balanced')
    lr.fit(X, y)
    nz = np.where(np.any(np.abs(lr.coef_) > 1e-8, axis=0))[0]
    print(f'  [{tag}] LASSO {se_mult}-SE: C={Cs[sel_i]:.5f} -> {len(nz)} feats')
    sel = [cols[i] for i in nz]
    coefs = lr.coef_[:, nz]
    cdf = pd.DataFrame({'Feature': sel, 'LASSO_Coef_class1': coefs[0],
                        'LASSO_Coef_class2': coefs[1],
                        'LASSO_Coef_class3': coefs[2]})
    cdf['AbsMax_Coef'] = cdf.iloc[:, 1:4].abs().max(axis=1)
    cdf = cdf.sort_values('AbsMax_Coef', ascending=False).reset_index(drop=True)
    return sel, cdf, {'C': float(Cs[sel_i]), 'se_mult': se_mult,
                      'cv_auc': float(mean_auc[sel_i]), 'n_cv': len(cols),
                      'n_sel': len(sel)}


def eval_model(lr, X, y, ds, all_cols, sel_cols, tag):
    idx = [all_cols.index(c) for c in sel_cols]
    rows, preds = [], []
    for d in DS_NAMES:
        m = ds == d
        pr = lr.predict_proba(X[m][:, idx])
        yh = pr.argmax(1); yd = y[m]
        aucs = [roc_auc_score((yd == c).astype(int), pr[:, c]) for c in range(3)]
        rows.append({'Arm': tag, 'Dataset': d, 'N': int(m.sum()),
                     'AUC_macro': np.mean(aucs), 'Accuracy': accuracy_score(yd, yh),
                     'Balanced_acc': balanced_accuracy_score(yd, yh),
                     'F1_macro': f1_score(yd, yh, average='macro', zero_division=0),
                     'RPS': rps_multi(yd, pr)})
        for i in range(len(yd)):
            preds.append({'Arm': tag, 'Dataset': d, 'ID': int(ids[m][i]),
                          'True_class': int(yd[i] + 1),
                          'Pred_class': int(yh[i] + 1),
                          'p_class1': pr[i, 0], 'p_class2': pr[i, 1],
                          'p_class3': pr[i, 2]})
    return pd.DataFrame(rows), pd.DataFrame(preds)


def load_icc_table():
    """icc_stability.csv was once re-saved as an xlsx container; handle both."""
    p = f'{HAB}/icc/icc_stability.csv'
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_excel(p, engine='openpyxl')


# ---------------------------------------------------------------- load
rad = pd.read_csv(f'{HAB}/features/05_habitat_radiomics_features_pop.csv')
icc = load_icc_table()
assign = pd.read_csv(f'{ROOT}/00_Split/00_dataset_assignments.csv')
for d in (rad, assign):
    d['ID'] = d['ID'].astype(int)
df = rad.merge(assign, on='ID', how='inner')
print(f'Merged: {df.shape}')

z = np.load(f'{HAB}/features/pop_kmeans_centers.npz')
print('Population centres (HU): ' + ', '.join(f'{c:.1f}' for c in z['centers']))


def is_minimal(c):
    return '_original_' in c or c.endswith('voxel_fraction')


icc_map = dict(zip(icc['Feature'], icc['ICC']))


def run_arm(pool_cols, tag):
    keep = [c for c in pool_cols
            if c in icc_map and np.isfinite(icc_map[c]) and icc_map[c] >= ICC_THRESH]
    funnel = {'pool': len(pool_cols), 'after_icc': len(keep)}
    print(f'\n[{tag}] ICC screen: {len(pool_cols)} -> {len(keep)}')
    tr = df['Dataset'] == 'Train'
    med = df.loc[tr, keep].median()
    df[keep] = df[keep].fillna(med)
    sd = df.loc[tr, keep].std()
    keep_v = sd[sd > 1e-6].index.tolist()
    mu = df.loc[tr, keep_v].mean(); sv = df.loc[tr, keep_v].std().replace(0, 1)
    X = ((df[keep_v] - mu) / sv).values
    cols = list(keep_v)
    y = df['Output'].values - 1
    tr_mask = df['Dataset'].values == 'Train'
    funnel['after_variance'] = len(cols)
    X, cols = reduce_redundant(X, cols, y, tr_mask, 0.90, tag)
    funnel['after_collinearity'] = len(cols)
    sel, cdf, linfo = lasso_select(X[tr_mask], y[tr_mask], cols, tag, 1.0)
    if len(sel) > MAX_SEL:
        sel, cdf, linfo = lasso_select(X[tr_mask], y[tr_mask], cols, tag, 2.0)
        if len(sel) > MAX_SEL:
            sel = cdf['Feature'].head(MAX_SEL).tolist()
    funnel['after_lasso'] = len(sel)
    idx = [cols.index(c) for c in sel]
    lr = LogisticRegression(solver='lbfgs',
                            max_iter=2000, class_weight='balanced')
    lr.fit(X[tr_mask][:, idx], y[tr_mask])
    met, pred = eval_model(lr, X, y, df['Dataset'].values,
                           cols, sel, tag)
    return met, pred, cdf, funnel | linfo, sel


pool_intra = [c for c in df.columns if c.startswith('intra_') and is_minimal(c)]
pool_both = [c for c in df.columns
             if (c.startswith('intra_') or c.startswith('peri_')) and is_minimal(c)]
y = df['Output'].values - 1
ds = df['Dataset'].values
ids = df['ID'].values

met_i, pred_i, cdf_i, fun_i, sel_i = run_arm(pool_intra, 'intra_pop')
met_b, pred_b, cdf_b, fun_b, sel_b = run_arm(pool_both, 'intra_peri_pop')

cdf_i.to_csv(f'{OUT}/02i_selected_features_intra_pop.csv', index=False)
cdf_b.to_csv(f'{OUT}/02i_selected_features_intra_peri_pop.csv', index=False)
pred_i.to_csv(f'{OUT}/02i_habitat_predictions_intra_pop.csv', index=False)
pred_b.to_csv(f'{OUT}/02i_habitat_predictions_intra_peri_pop.csv', index=False)
metrics = pd.concat([met_i, met_b], ignore_index=True)
metrics.round(4).to_csv(f'{OUT}/02i_habitat_pop_metrics.csv', index=False)
pd.DataFrame([fun_i, fun_b]).to_csv(f'{OUT}/02i_feature_funnel.csv', index=False)

print('\nmacro AUROC (population clustering):')
print(metrics.pivot(index='Dataset', columns='Arm', values='AUC_macro')
      .round(4).to_string())

# ------------------------------------------------- paradigm comparison + boot
pp_std = pd.read_csv(f'{HAB}/std/02g_habitat_predictions_std.csv')
pp_ext = pd.read_csv(f'{HAB}/std/02g_habitat_predictions_extended.csv')
PAIRS = {'intra': (pred_i, pp_std), 'intra_peri': (pred_b, pp_ext)}
rows = []
for tag, (pnew, pold) in PAIRS.items():
    mrg = pnew.merge(pold, on=['ID', 'Dataset'], suffixes=('_pop', '_pp'))
    for d in DS_NAMES:
        s = mrg[mrg.Dataset == d]
        yv = s.True_class_pop.values - 1
        Pn = s[['p_class1_pop', 'p_class2_pop', 'p_class3_pop']].values
        Po = s[['p_class1_pp', 'p_class2_pp', 'p_class3_pp']].values

        def mauc(P, yv, idx):
            return np.mean([roc_auc_score((yv[idx] == c).astype(int), P[idx, c])
                            for c in range(3)])
        n = len(s)
        diffs = []
        for _ in range(N_BOOT):
            idx = RNG.randint(0, n, n)
            diffs.append(mauc(Pn, yv, idx) - mauc(Po, yv, idx))
        diffs = np.array(diffs)
        d_obs = mauc(Pn, yv, np.arange(n)) - mauc(Po, yv, np.arange(n))
        p_two = 2 * min((diffs < 0).mean(), (diffs > 0).mean())
        rows.append({'Arm': tag, 'Dataset': d, 'N': n,
                     'AUC_pop': d_obs + mauc(Po, yv, np.arange(n)),
                     'AUC_per_patient': mauc(Po, yv, np.arange(n)),
                     'diff': d_obs,
                     'CI2.5': np.percentile(diffs, 2.5),
                     'CI97.5': np.percentile(diffs, 97.5),
                     'p_two_sided': max(p_two, 1 / N_BOOT)})
boot = pd.DataFrame(rows)
boot.round(4).to_csv(f'{OUT}/02i_paradigm_bootstrap.csv', index=False)
print('\nPaired bootstrap (pop - per-patient), macro AUROC:')
print(boot.round(4).to_string(index=False))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=DPI)
x = np.arange(4); w = 0.2
arms = [('intra', pp_std, pred_i, '#4C72B0', '#9FB8D8'),
        ('intra_peri', pp_ext, pred_b, '#C44E52', '#E5AFAA')]
for j, (tag, pold, pnew, c1, c2) in enumerate(arms):
    old_auc = [pold[pold.Dataset == d].pipe(
        lambda s: np.mean([roc_auc_score((s.True_class.values - 1 == c).astype(int),
                                         s[['p_class1', 'p_class2', 'p_class3']]
                                         .values[:, c]) for c in range(3)]))
        for d in DS_NAMES]
    new_auc = boot[boot.Arm == tag].sort_values('Dataset').AUC_pop.values
    ax.bar(x - w / 2 + j * w, old_auc, w, color=c1,
           label=f'Per-patient clustering ({tag})')
    ax.bar(x + w / 2 + j * w, new_auc, w, color=c2,
           label=f'Population clustering ({tag})')
    for xi, v in zip(x - w / 2 + j * w, old_auc):
        ax.text(xi, v + 0.004, f'{v:.3f}', ha='center', fontsize=7.5)
    for xi, v in zip(x + w / 2 + j * w, new_auc):
        ax.text(xi, v + 0.004, f'{v:.3f}', ha='center', fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels([DS_DISPLAY[d] for d in DS_NAMES], fontsize=10)
ax.set_ylabel('macro AUROC'); ax.set_ylim(0.6, 0.9)
ax.set_title('Habitat clustering paradigm: per-patient vs population-level (k=3)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=0.3, axis='y')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT}/02i_paradigm_comparison.{ext}', dpi=DPI,
                bbox_inches='tight')
plt.close(fig)

# habitat availability audit
fr_cols = [c for c in df.columns if c.endswith('voxel_fraction')]
audit = (df[fr_cols] == 0).mean().round(4)
audit.to_csv(f'{OUT}/02i_empty_habitat_audit.csv')

with open(f'{OUT}/02i_population_summary.txt', 'w', encoding='utf-8') as f:
    f.write('POPULATION-LEVEL HABITAT CLUSTERING (k=3) - SENSITIVITY ARM\n')
    f.write('=' * 60 + '\n')
    f.write(f"Centres (HU cold->dense): {np.round(z['centers'], 1).tolist()}\n")
    f.write(f"Fixed boundaries (HU): {np.round(z['bounds'], 1).tolist()}\n")
    f.write('Centres learned on pooled Train voxels only (no leakage);\n'
            'applied to all patients with fixed thresholds.\n\n')
    f.write(f"Funnel intra_pop     : {fun_i}\n")
    f.write(f"Funnel intra_peri_pop: {fun_b}\n\n")
    f.write('macro AUROC:\n' +
            metrics.pivot(index='Dataset', columns='Arm', values='AUC_macro')
            .round(4).to_string())
    f.write('\n\nPaired bootstrap vs per-patient arm:\n' +
            boot.round(4).to_string(index=False))
    f.write('\n\nEmpty-habitat rates (pop clustering):\n' + audit.to_string())
    f.write('\n\nFinal features (intra_pop):\n' +
            cdf_i[['Feature', 'AbsMax_Coef']].round(4).to_string(index=False))
    f.write('\n\nFinal features (intra_peri_pop):\n' +
            cdf_b[['Feature', 'AbsMax_Coef']].round(4).to_string(index=False))
print('\nDONE ->', OUT)
