"""
RERUN_FULL - shared post-hoc temperature scaling for the 5:3:2 fusion
=====================================================================
Single source of truth for the calibration step of the deployed late-fusion
model (weighted soft voting, clinical : deep-learning : habitat = 5 : 3 : 2).

WHY THIS MODULE EXISTS
  The main line (04_fusion.py) reports the deployed fusion model AFTER a
  post-hoc temperature scaling step: a single temperature T is grid-searched
  on the Training set to minimise the ranked probability score (RPS), then
  applied to the raw soft-voting probabilities before every downstream
  metric / figure. 04b_weight_sensitivity.py and 04c_ablation.py must report
  macro AUROC / RPS in EXACTLY the same probability space, otherwise the
  same 5:3:2 configuration would show two different AUROC values across
  figures (raw 0.8649 vs temperature-scaled 0.8625).

  Hence both scripts import this module and call
  ``temperature_scale_probs(Pf, T)`` with the SAME cached T. Refitting T per
  weight combination is neither necessary nor desirable: T is a fixed,
  pre-specified post-processing step of the pipeline, and holding it constant
  is what makes the weight/ablation comparisons fair.

  The fitted T is cached by 04_fusion.py in
  ``04_Fusion/fusion_temperature.json``.

Note that temperature scaling is a monotone transform of the per-sample
logits: it leaves argmax(P) - and therefore accuracy, F1 and the confusion
matrices - unchanged, while affecting probability-dependent quantities
(one-vs-rest AUROC slightly, RPS and calibration noticeably).
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPERATURE_JSON = os.path.join(_HERE, 'fusion_temperature.json')

T_GRID = np.linspace(0.4, 4.0, 73)      # identical to 04_fusion.py
T_FIT_REF = 'Train'                      # fitted on the training split only


def logits_from_probs(Pm, eps=1e-9):
    """Recover logits with class 1 as reference: z1=0, z2=log(p2/p1), z3=log(p3/p1)."""
    Pm = np.clip(np.asarray(Pm, dtype=float), eps, 1 - eps)
    Pm = Pm / Pm.sum(axis=1, keepdims=True)
    z = np.zeros((len(Pm), 3))
    z[:, 1] = np.log(Pm[:, 1] / Pm[:, 0])
    z[:, 2] = np.log(Pm[:, 2] / Pm[:, 0])
    return z


def temperature_scale_probs(Pm, T, eps=1e-9):
    """Sharpen / soften class probabilities by a single temperature T."""
    z = logits_from_probs(Pm, eps) / float(T)
    ez = np.exp(z - z.max(axis=1, keepdims=True))
    return ez / ez.sum(axis=1, keepdims=True)


def rps3(y, Pm):
    """Ranked probability score for a 3-class ordinal problem (lower better)."""
    Pm = np.asarray(Pm, dtype=float)
    yoh = np.zeros_like(Pm)
    yoh[np.arange(len(y)), y] = 1
    cp, cy = np.cumsum(Pm, 1), np.cumsum(yoh, 1)
    return float(np.mean(np.sum((cp - cy) ** 2, 1) / (Pm.shape[1] - 1)))


def fit_temperature(Pm, y, ds, ref=T_FIT_REF):
    """Grid-search the RPS-minimising temperature on the reference split."""
    m = np.asarray(ds) == ref
    if m.sum() == 0:
        m = np.ones(len(y), dtype=bool)
    best, best_T = 1e9, 1.0
    for T in T_GRID:
        r = rps3(np.asarray(y)[m], temperature_scale_probs(np.asarray(Pm)[m], T))
        if r < best:
            best, best_T = r, T
    return float(best_T)


def deployed_calibration(Pf, weights, T, eps=1e-9):
    """Post-hoc calibration as used by the deployed fusion model.

    Temperature scaling corrects the systematic *under-confidence* introduced
    by averaging several probability vectors; a single component has no such
    averaging and therefore needs no correction.  Accordingly the scaling is
    applied only to blends of >= 2 modalities, while single-modality
    configurations are returned unchanged so that they coincide exactly with
    the component models reported in Figures 3-4.

    weights : (w_clinical, w_deep_learning, w_habitat)
    """
    n_comp = sum(1 for w in weights if w > eps)
    if n_comp >= 2:
        return temperature_scale_probs(Pf, T)
    return np.asarray(Pf, dtype=float)


def save_temperature(T, path=TEMPERATURE_JSON, extra=None):
    payload = {
        'T': float(T),
        'fitted_on': T_FIT_REF,
        'criterion': 'RPS',
        'grid': [float(T_GRID[0]), float(T_GRID[-1]), int(len(T_GRID))],
        'note': ('post-hoc calibration of the 5:3:2 weighted soft-voting '
                 'fusion; applied unchanged to the weight-sensitivity and '
                 'ablation analyses (04b / 04c)'),
    }
    if extra:
        payload.update(extra)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return path


def load_temperature(path=TEMPERATURE_JSON):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found - run 04_fusion.py first; it caches the '
            f'fitted temperature for 04b / 04c.')
    with open(path, encoding='utf-8') as f:
        return float(json.load(f)['T'])
