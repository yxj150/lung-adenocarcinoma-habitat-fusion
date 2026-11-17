"""
RERUN_FULL - Step 1: Clinical baseline model (multi-algorithm comparison)
========================================================================
Input : RERUN_FULL/00_Split/00_clinical_all.csv  (from F:/yu/DATE)
Workflow (training set only for every selection step):
  1a. Univariate screening (chi-square / Kruskal-Wallis, p<0.05)
  1b. Multinomial LASSO (l1, 5-fold CV, 1-SE rule)
  2.  Four algorithms with 5-fold stratified CV tuning on Train:
      Logistic regression / SVM(RBF) / Random forest / XGBoost
  3.  Evaluation on Train / Internal_Val / External_Val_1 / External_Val_2
      ROC (macro OvR + per class), confusion matrix, calibration,
      decision-curve analysis, SHAP (XGBoost)
  4.  Best model = highest Internal_Val macro AUC -> saved for Step 4 fusion
Output -> F:/yu/RERUN_FULL/01_Clinical/
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix,
                             accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)
import xgboost as xgb
import shap
from joblib import Parallel, delayed

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '8')   # avoid CPU-probe hangs
warnings.filterwarnings('ignore')

BASE = 'F:/yu'
SPLIT = f'{BASE}/RERUN_FULL/00_Split'
OUT = f'{BASE}/RERUN_FULL/01_Clinical'
os.makedirs(OUT, exist_ok=True)

DPI = 600
CLASS_NAMES = ['class1', 'class2', 'class3']
DS_NAMES = ['Train', 'Internal_Val', 'External_Val_1', 'External_Val_2']
DS_DISPLAY = {'Train': 'Training set', 'Internal_Val': 'Internal validation',
              'External_Val_1': 'External validation 1',
              'External_Val_2': 'External validation 2'}
MODEL_NAMES = ['LR', 'SVM', 'RF', 'XGBoost']
MODEL_DISPLAY = {'LR': 'Logistic regression', 'SVM': 'SVM (RBF)',
                 'RF': 'Random forest', 'XGBoost': 'XGBoost'}
MODEL_COLORS = {'LR': '#4C72B0', 'SVM': '#DD8452', 'RF': '#55A868',
                'XGBoost': '#C44E52'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.5,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 6.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def save_fig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}.pdf / .png')


# ------------------------------------------------------------------ load
df = pd.read_csv(f'{SPLIT}/00_clinical_all.csv')
FEATS_ALL = [c for c in df.columns if c not in ('ID', 'Dataset', 'Output')]
train = df[df['Dataset'] == 'Train'].reset_index(drop=True)
ytr = train['Output'].values - 1
print(f'Loaded {len(df)} patients | Train {len(train)} | '
      f'{len(FEATS_ALL)} variables')

# ------------------------------------------------------------------ 1a univariate
uni_rows = []
for f in FEATS_ALL:
    x = train[f].values
    groups = [x[ytr == k] for k in range(3)]
    if len(np.unique(x)) <= 10:
        ct = pd.crosstab(train[f], train['Output'])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        test, stat = 'chi-square', chi2
    else:
        h, p = stats.kruskal(*groups)
        test, stat = 'Kruskal-Wallis', h
    uni_rows.append({'Feature': f, 'Test': test, 'Statistic': round(stat, 3),
                     'P': p, 'Selected_P<0.05': p < 0.05})
uni = pd.DataFrame(uni_rows)
uni.to_csv(f'{OUT}/01_univariate_screening.csv', index=False)
feats_uni = uni.loc[uni['Selected_P<0.05'], 'Feature'].tolist()
print(f'Univariate screening: {len(feats_uni)}/{len(FEATS_ALL)} features p<0.05')

# ------------------------------------------------------------------ 1b LASSO
scaler = StandardScaler().fit(train[feats_uni])
Xtr_sc = scaler.transform(train[feats_uni])
Cs = np.logspace(-3, 1, 9)
cv5 = StratifiedKFold(5, shuffle=True, random_state=42)


def _lasso_auc(C, tr_i, va_i):
    lasso = LogisticRegression(penalty='l1', C=C, solver='saga', max_iter=3000,
                               random_state=42, class_weight='balanced')
    lasso.fit(Xtr_sc[tr_i], ytr[tr_i])
    if not np.any(lasso.coef_):
        return 0.5
    p = lasso.predict_proba(Xtr_sc[va_i])
    return roc_auc_score(ytr[va_i], p, multi_class='ovr', average='macro')


fold_splits = list(cv5.split(Xtr_sc, ytr))
auc_mat = np.array(Parallel(n_jobs=5)(
    delayed(_lasso_auc)(C, tr_i, va_i) for C in Cs for (tr_i, va_i) in fold_splits)
).reshape(len(Cs), 5)
mean_auc = auc_mat.mean(1)
se_auc = auc_mat.std(1) / np.sqrt(5)
best_i = int(np.argmax(mean_auc))
best_C = Cs[best_i]
thr = mean_auc[best_i] - se_auc[best_i]
sel_i = int(min([i for i in range(len(Cs)) if mean_auc[i] >= thr],
                key=lambda i: Cs[i]))
C_1se = Cs[sel_i]

lasso_final = LogisticRegression(penalty='l1', C=C_1se, solver='saga',
                                 max_iter=8000, random_state=42,
                                 class_weight='balanced').fit(Xtr_sc, ytr)
coef_df = pd.DataFrame({'Feature': feats_uni,
                        'coef_class1': lasso_final.coef_[0],
                        'coef_class2': lasso_final.coef_[1],
                        'coef_class3': lasso_final.coef_[2]})
coef_df['n_nonzero_classes'] = (coef_df[['coef_class1', 'coef_class2',
                                         'coef_class3']] != 0).sum(axis=1)
feats_lasso = coef_df.loc[coef_df['n_nonzero_classes'] > 0, 'Feature'].tolist()
coef_df.to_csv(f'{OUT}/01_lasso_coefficients.csv', index=False)
print(f'LASSO (C={C_1se:.3f}, 1-SE of best {best_C:.3f}): '
      f'{len(feats_lasso)} features retained')
pd.DataFrame({'C': Cs, 'mean_auc': mean_auc, 'se': se_auc}).to_csv(
    f'{OUT}/01_lasso_cv_curve.csv', index=False)

mm = 1 / 25.4
fig, ax = plt.subplots(figsize=(60 * mm, 60 * mm))
ax.errorbar(np.log10(Cs), mean_auc, yerr=se_auc, fmt='o-', ms=2.5, lw=1,
            color='#4C72B0', capsize=2, elinewidth=0.7)
ax.axvline(np.log10(best_C), color='gray', ls='--', lw=0.7)
ax.axvline(np.log10(C_1se), color='gray', ls=':', lw=0.7)
ax.set_xlabel('log10(C)  (smaller = stronger penalty)')
ax.set_ylabel('Mean macro AUC (5-fold CV)')
ax.set_ylim(0.5, 1.0)
save_fig(fig, '01_lasso_cv_curve')

FEATS = feats_lasso if len(feats_lasso) >= 3 else feats_uni
print(f'Final feature set ({len(FEATS)}): {FEATS}')

Xs, ys, ids = {}, {}, {}
for ds in DS_NAMES:
    sub = df[df['Dataset'] == ds].reset_index(drop=True)
    Xs[ds] = sub[FEATS].values.astype(float)
    ys[ds] = sub['Output'].values - 1
    ids[ds] = sub['ID'].values

cw = {k: len(ytr) / (3 * np.sum(ytr == k)) for k in range(3)}
sw_tr = np.array([cw[y] for y in ytr])

# ------------------------------------------------------------------ 2 models
def fit_models():
    models = {}
    pipe = Pipeline([('sc', StandardScaler()),
                     ('lr', LogisticRegression(max_iter=10000,
                                               class_weight='balanced'))])
    gs = GridSearchCV(pipe, {'lr__C': [0.01, 0.1, 1, 10]}, cv=cv5,
                      scoring='roc_auc_ovr', n_jobs=4)
    gs.fit(train[FEATS], ytr)
    models['LR'] = gs.best_estimator_

    pipe = Pipeline([('sc', StandardScaler()),
                     ('svc', SVC(probability=True, class_weight='balanced'))])
    gs = GridSearchCV(pipe, {'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 0.1]},
                      cv=cv5, scoring='roc_auc_ovr', n_jobs=4)
    gs.fit(train[FEATS], ytr)
    models['SVM'] = gs.best_estimator_

    gs = GridSearchCV(RandomForestClassifier(random_state=42,
                                             class_weight='balanced'),
                      {'n_estimators': [500], 'max_depth': [6],
                       'min_samples_leaf': [5], 'max_features': ['sqrt']},
                      cv=cv5, scoring='roc_auc_ovr', n_jobs=4)
    gs.fit(train[FEATS], ytr)
    models['RF'] = gs.best_estimator_

    gs = GridSearchCV(
        xgb.XGBClassifier(objective='multi:softprob', num_class=3,
                          n_estimators=400, subsample=0.8, colsample_bytree=0.8,
                          random_state=42, eval_metric='mlogloss',
                          tree_method='hist'),
        {'max_depth': [3, 6], 'learning_rate': [0.02, 0.05, 0.1],
         'min_child_weight': [1, 3]},
        cv=cv5, scoring='roc_auc_ovr', n_jobs=4)
    try:
        gs.fit(train[FEATS], ytr, sample_weight=sw_tr)
    except TypeError:
        gs.fit(train[FEATS], ytr)
    models['XGBoost'] = gs.best_estimator_
    return models


models = fit_models()
with open(f'{OUT}/01_best_hyperparameters.txt', 'w', encoding='utf-8') as f:
    for m in MODEL_NAMES:
        est = models[m]
        if hasattr(est, 'best_params_'):
            f.write(f'{MODEL_DISPLAY[m]}: {est.best_params_}\n')
        elif hasattr(est, 'steps'):
            f.write(f'{MODEL_DISPLAY[m]}: ' + '; '.join(
                f'{n}={v}' for n, v in est.steps[1][1].get_params().items()
                if n in ('C', 'gamma', 'kernel')) + '\n')
print('Hyperparameter tuning finished.')

# ------------------------------------------------------------------ 3 predictions
preds, pred_rows = {}, []
for m in MODEL_NAMES:
    preds[m] = {}
    for ds in DS_NAMES:
        pr = models[m].predict_proba(Xs[ds])
        preds[m][ds] = pr
        yh = pr.argmax(1)
        for i in range(len(yh)):
            pred_rows.append({'Model': m, 'Dataset': ds, 'ID': ids[ds][i],
                              'True_class': ys[ds][i] + 1,
                              'Predicted_class': yh[i] + 1,
                              'p_class1': pr[i, 0], 'p_class2': pr[i, 1],
                              'p_class3': pr[i, 2]})
pred_all = pd.DataFrame(pred_rows)
pred_all.to_csv(f'{OUT}/01_predictions_all_models.csv', index=False)
for m in MODEL_NAMES:
    pred_all[pred_all['Model'] == m].drop(columns='Model').to_csv(
        f'{OUT}/01_predictions_{m}.csv', index=False)

# ------------------------------------------------------------------ 4 metrics
met_rows = []
for m in MODEL_NAMES:
    for ds in DS_NAMES:
        pr, y = preds[m][ds], ys[ds]
        yh = pr.argmax(1)
        aucs_c = [roc_auc_score((y == k).astype(int), pr[:, k]) for k in range(3)]
        met_rows.append({
            'Model': m, 'Dataset': ds, 'N': len(y),
            'AUC_macro': np.mean(aucs_c),
            'AUC_class1': aucs_c[0], 'AUC_class2': aucs_c[1],
            'AUC_class3': aucs_c[2],
            'Accuracy': accuracy_score(y, yh),
            'Balanced_acc': balanced_accuracy_score(y, yh),
            'Precision_macro': precision_score(y, yh, average='macro',
                                               zero_division=0),
            'Recall_macro': recall_score(y, yh, average='macro',
                                         zero_division=0),
            'F1_macro': f1_score(y, yh, average='macro', zero_division=0)})
metrics = pd.DataFrame(met_rows)
for c in metrics.columns[3:]:
    metrics[c] = metrics[c].round(4)
metrics.to_csv(f'{OUT}/01_model_metrics.csv', index=False)
print(metrics[['Model', 'Dataset', 'AUC_macro', 'AUC_class3']].to_string(
    index=False))

iv = metrics[metrics['Dataset'] == 'Internal_Val']
BEST = iv.loc[iv['AUC_macro'].idxmax(), 'Model']
print(f'>>> Best clinical model (Internal_Val macro AUC): {BEST}')

# best-model predictions as the canonical clinical component
pred_all[pred_all['Model'] == BEST].drop(columns='Model').to_csv(
    f'{OUT}/01_predictions_best.csv', index=False)

# ------------------------------------------------------------------ DeLong
def delong_pair(y, pa, pb, k):
    y01 = (y == k).astype(int)
    n1, n0 = int(y01.sum()), int((1 - y01).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, np.nan

    def midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j + 1)
            i = j
        T2 = np.empty(N, float)
        T2[J] = T
        return T2

    def auc_var(p):
        tab1 = y01 == 1
        X = np.where(tab1, p, p)
        TX = midrank(np.where(tab1, p, -p))
        T01 = midrank(np.where(tab1, 1, 0))
        A1 = (TX[tab1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        A2 = (TX[~tab1].sum() - n0 * (n0 + 1) / 2) / (n1 * n0)
        V = (A1 - A2) / 2
        S1 = (TX[tab1].sum() - n1 * (n1 + 1) / 2 - n1 * n0 * A1) ** 2 / \
             ((n1 - 1) * (n0 - 1) * n1 * n0)
        S2 = (TX[~tab1].sum() - n0 * (n0 + 1) / 2 - n1 * n0 * A2) ** 2 / \
             ((n1 - 1) * (n0 - 1) * n1 * n0)
        S = S1 + S2 - 2 * V
        return A1, S / (n1 * n0)

    def auc_var_simple(p):
        """DeLong variance via midrank of the score itself."""
        TX = midrank(p)
        T01 = midrank(y01)
        A1 = (TX[y01 == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        A2 = (TX[y01 == 0].sum() - n0 * (n0 + 1) / 2) / (n1 * n0)
        V10 = (TX[y01 == 1] - A1)
        V01 = (TX[y01 == 0] - A2)
        var = (np.var(V10, ddof=1) / n1 if n1 > 1 else 0) + \
              (np.var(V01, ddof=1) / n0 if n0 > 1 else 0)
        return A1, var

    a, va = auc_var_simple(pa[:, k])
    b, vb = auc_var_simple(pb[:, k])
    z = (a - b) / np.sqrt(max(va + vb, 1e-12))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return a, b, p


dl_rows = []
for ds in DS_NAMES:
    for m in MODEL_NAMES:
        if m == BEST:
            continue
        for k in range(3):
            ra, rb, p = delong_pair(ys[ds], preds[BEST][ds], preds[m][ds], k)
            dl_rows.append({'Dataset': ds, 'Comparison': f'{BEST} vs {m}',
                            'Class': CLASS_NAMES[k], f'AUC_{BEST}': ra,
                            f'AUC_{m}': rb, 'P': p})
delong = pd.DataFrame(dl_rows)
for c in delong.columns:
    if c.startswith('AUC') or c == 'P':
        delong[c] = delong[c].round(4)
delong.to_csv(f'{OUT}/01_delong_best_vs_others.csv', index=False)

# ------------------------------------------------------------------ 5 figures
fig, axes = plt.subplots(2, 2, figsize=(120 * mm, 120 * mm))
for ax, ds in zip(axes.ravel(), DS_NAMES):
    y = ys[ds]
    for m in MODEL_NAMES:
        pr = preds[m][ds]
        fprs = np.linspace(0, 1, 200)
        tprs, aucs_c = [], []
        for k in range(3):
            fpr, tpr, _ = roc_curve((y == k).astype(int), pr[:, k])
            tprs.append(np.interp(fprs, fpr, tpr))
            aucs_c.append(auc(fpr, tpr))
        ax.plot(fprs, np.mean(tprs, 0), color=MODEL_COLORS[m], lw=1.1,
                label=f'{m} (AUC={np.mean(aucs_c):.3f})')
    ax.plot([0, 1], [0, 1], color='gray', ls='--', lw=0.6)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
    ax.set_title(DS_DISPLAY[ds]); ax.legend(loc='lower right', frameon=False)
fig.suptitle('ROC comparison of four machine-learning models (macro OvR)',
             fontsize=9, y=1.0)
fig.tight_layout()
save_fig(fig, '01_roc_4models_4datasets')

fig, axes = plt.subplots(3, 4, figsize=(120 * mm * 2, 3 * 60 * mm))
for r, k in enumerate(range(3)):
    for c_, ds in enumerate(DS_NAMES):
        ax = axes[r, c_]
        y01 = (ys[ds] == k).astype(int)
        pr = preds[BEST][ds][:, k]
        fpr, tpr, _ = roc_curve(y01, pr)
        ax.plot(fpr, tpr, color=MODEL_COLORS[BEST], lw=1.1,
                label=f'AUC={auc(fpr, tpr):.3f}')
        ax.plot([0, 1], [0, 1], color='gray', ls='--', lw=0.6)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
        ax.set_title(f'{CLASS_NAMES[k]} - {DS_DISPLAY[ds]}', fontsize=7)
        ax.legend(loc='lower right', frameon=False)
fig.tight_layout()
save_fig(fig, f'01_roc_3class_{BEST}')

fig, axes = plt.subplots(2, 4, figsize=(4 * 60 * mm, 2 * 55 * mm))
for c_, ds in enumerate(DS_NAMES):
    y = ys[ds]
    yh = preds[BEST][ds].argmax(1)
    cm = confusion_matrix(y, yh, labels=[0, 1, 2])
    cmn = cm / cm.sum(1, keepdims=True)
    for r, (mat, fmt) in enumerate([(cm, 'd'), (cmn, '.2f')]):
        ax = axes[r, c_]
        ax.imshow(mat, cmap='Blues', vmin=0, vmax=cm.max())
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}' if r else f'{v}', ha='center',
                        va='center', fontsize=6,
                        color='white' if v > cm.max() * 0.6 else 'black')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True' if c_ == 0 else '')
        ax.set_title(f'{DS_DISPLAY[ds]}', fontsize=6.5)
fig.suptitle(f'Confusion matrices - {MODEL_DISPLAY[BEST]} '
             '(top: counts, bottom: row-normalized)', fontsize=8)
fig.tight_layout()
save_fig(fig, f'01_confusion_{BEST}')

fig, axes = plt.subplots(3, 4, figsize=(120 * mm * 2, 3 * 60 * mm))
hl_rows = []
for r, k in enumerate(range(3)):
    for c_, ds in enumerate(DS_NAMES):
        ax = axes[r, c_]
        y01 = (ys[ds] == k).astype(int)
        pr = preds[BEST][ds][:, k]
        hl_p = np.nan
        try:
            q = pd.qcut(pr, 10, duplicates='drop')
            grp = pd.DataFrame({'p': pr, 'y': y01, 'q': q}).groupby(
                'q', observed=True)
            obs = grp['y'].sum(); exp = grp['p'].sum(); nn = grp['y'].count()
            hl = ((obs - exp) ** 2 / (exp * (1 - exp / nn) + 1e-12)).sum()
            hl_p = 1 - stats.chi2.cdf(hl, len(obs) - 1)
        except Exception:
            pass
        brier = np.mean((pr - y01) ** 2)
        hl_rows.append({'Dataset': ds, 'Class': CLASS_NAMES[k],
                        'Brier': round(brier, 4),
                        'HL_Chi2_P': round(hl_p, 4) if hl_p == hl_p else np.nan})
        if len(np.unique(y01)) == 2 and y01.sum() > 3:
            fop, mpp = calibration_curve(y01, pr, n_bins=10, strategy='uniform')
            ax.plot(mpp, fop, 'o-', ms=2.5, lw=1, color=MODEL_COLORS[BEST])
        ax.plot([0, 1], [0, 1], color='gray', ls='--', lw=0.6)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed rate')
        ax.set_title(f'{CLASS_NAMES[k]} - {DS_DISPLAY[ds]}\nBrier={brier:.3f}',
                     fontsize=6.5)
fig.tight_layout()
save_fig(fig, f'01_calibration_{BEST}')
pd.DataFrame(hl_rows).to_csv(f'{OUT}/01_calibration_brier_hl.csv', index=False)


def net_benefit(y01, pr, thr):
    tp = np.sum((pr >= thr) & (y01 == 1))
    fp = np.sum((pr >= thr) & (y01 == 0))
    return tp / len(y01) - fp / len(y01) * (thr / max(1 - thr, 1e-9))


fig, axes = plt.subplots(3, 4, figsize=(120 * mm * 2, 3 * 60 * mm))
thrs = np.linspace(0.02, 0.98, 40)
for r, k in enumerate(range(3)):
    for c_, ds in enumerate(DS_NAMES):
        ax = axes[r, c_]
        y01 = (ys[ds] == k).astype(int)
        pr = preds[BEST][ds][:, k]
        nb = np.array([net_benefit(y01, pr, t) for t in thrs])
        prev = y01.mean()
        ax.plot(thrs, nb, color=MODEL_COLORS[BEST], lw=1.2, label=BEST)
        ax.plot(thrs, [prev - (1 - prev) * (t / max(1 - t, 1e-9)) for t in thrs],
                color='gray', ls=':', lw=0.9, label='Treat all')
        ax.axhline(0, color='gray', ls='-', lw=0.5)
        ax.set_ylim(-0.1, max(0.05, np.nanmax(nb) * 1.1))
        ax.set_xlim(0, 1)
        ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
        ax.set_title(f'{CLASS_NAMES[k]} - {DS_DISPLAY[ds]}', fontsize=6.5)
        ax.legend(loc='upper right', frameon=False, fontsize=6)
fig.tight_layout()
save_fig(fig, f'01_dca_{BEST}')

try:
    xgb_model = models['XGBoost']
    explainer = shap.TreeExplainer(xgb_model)
    Xpd = train[FEATS].astype(float)
    sv = explainer.shap_values(Xpd)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    if np.asarray(sv).ndim == 2:
        sv = np.stack([sv, -sv, sv], axis=-1)
    imp = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(imp)
    fig, axes = plt.subplots(1, 3, figsize=(3 * 60 * mm, 70 * mm))
    for k, ax in enumerate(axes):
        ax.barh(range(len(order)), imp[order], color=MODEL_COLORS['XGBoost'],
                lw=0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([FEATS[i] for i in order], fontsize=5.5)
        ax.set_xlabel('mean |SHAP|')
        ax.set_title(f'{CLASS_NAMES[k]} (global)', fontsize=7)
    fig.tight_layout()
    save_fig(fig, '01_shap_global_importance')
    shap_tab = pd.DataFrame({'Feature': FEATS,
                             'meanSHAP_class1': np.abs(sv[:, :, 0]).mean(0),
                             'meanSHAP_class2': np.abs(sv[:, :, 1]).mean(0),
                             'meanSHAP_class3': np.abs(sv[:, :, 2]).mean(0)})
    shap_tab['meanSHAP_overall'] = np.abs(sv).mean(axis=(0, 2))
    shap_tab.sort_values('meanSHAP_overall', ascending=False).round(4).to_csv(
        f'{OUT}/01_shap_importance.csv', index=False)
except Exception as e:
    print('SHAP failed:', e)

# ------------------------------------------------------------------ 6 summary
with open(f'{OUT}/01_clinical_model_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f"""CLINICAL BASELINE MODEL - RERUN_FULL
====================================
Cohort: F:/yu/DATE (4 pre-defined splits, {len(df)} patients)
Train {len(ys['Train'])} / Internal_Val {len(ys['Internal_Val'])} /
External_Val_1 {len(ys['External_Val_1'])} / External_Val_2 {len(ys['External_Val_2'])}

Feature selection (training set only):
- Univariate screening: {len(feats_uni)}/{len(FEATS_ALL)} variables p<0.05
- Multinomial LASSO (C={C_1se:.3f}, 1-SE rule): {len(feats_lasso)} retained
- Final feature set ({len(FEATS)}): {', '.join(FEATS)}

Algorithms: LR / SVM(RBF) / RF / XGBoost, 5-fold stratified CV tuning.
Best model (Internal_Val macro AUC): {BEST}

Macro AUC (OvR) per dataset:
""")
    piv = metrics.pivot(index='Model', columns='Dataset', values='AUC_macro')
    f.write(piv.round(4).to_string() + f'\n\nBest model: {BEST}\n')
    f.write('\nFull metrics table (best model):\n')
    f.write(metrics[metrics.Model == BEST].to_string(index=False) + '\n')
print('DONE ->', OUT)
