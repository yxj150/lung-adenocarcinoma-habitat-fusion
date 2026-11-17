# Code release: final trained models (clinical, habitat, deep learning, fusion)

This repository contains the code and final model artefacts for the four
models reported in the manuscript, so that future investigators can apply
the exact deployed models:

1. **Clinical model** — Random Forest on 7 clinical/CT descriptors.
2. **Habitat (radiomics) model** — population-level 3-habitat decomposition
   (k-means centres learned once on pooled training voxels) + 14-feature
   multinomial LASSO.
3. **Deep-learning model** — Kinetics-400-pretrained ResNet18-3D,
   five-seed ensemble (seeds 42 / 123 / 2024 / 3407 / 777) with 8-flip
   test-time augmentation, on 64³ tumour-centred crops.
4. **Fusion model** — late fusion of the three components by weighted soft
   voting (clinical : deep learning : habitat = 5 : 3 : 2) followed by a
   post-hoc temperature scaling (T = 0.45, fitted on the training set).

Cohort/dataset split assignment tables are part of the study data, not the
code, and are therefore not included; every script reads them from a single
`00_Split` CSV (see each script header).

---

## Repository layout

```
clinical/
  clinical_model.py                     full clinical pipeline: univariate
                                        screening -> multinomial LASSO (1-SE)
                                        -> LR / SVM / RF / XGBoost with 5-fold
                                        CV -> best model selection + SHAP
  models/final_hyperparameters.txt      final RF configuration (deployed)
  models/final_features.txt             final 7-feature set

habitat/
  habitat_feature_extraction.py         population-level habitat clustering
                                        (KMeans k=3 on pooled TRAIN tumour
                                        voxels, random_state=0) + PyRadiomics
                                        feature extraction (intra + 3-mm ring)
  habitat_model_14features.py           the deployed 14-feature habitat model:
                                        ICC screen -> z-score -> Spearman
                                        |rho|<0.9 -> multinomial LASSO (10-fold,
                                        1-SE) -> multinomial LR; reproduces the
                                        final features, coefficients and metrics
  models/pop_kmeans_centers.npz         final k=3 centres (-729.0 / -411.7 /
                                        -3.2 HU) and boundaries (-570.3 /
                                        -207.4 HU)
  models/final_14feature_lasso_coefficients.csv
                                        final 14 features + per-class LASSO
                                        coefficients (the "final features and
                                        weights" of the habitat model)
  models/02i_population_summary.txt     reference metrics of the habitat model

deep_learning/
  preprocess_64cube.py                  tumour-centred 64³ crop extraction,
                                        HU clip [-1000, 400] -> [0, 1]
  train_r3d18.py                        model definitions (Net), training loop
                                        and evaluation for the DL branch
  inference_reference_gradcampp.py      reference inference code: loads the
                                        five checkpoints, rebuilds the Net,
                                        computes ensemble probabilities and
                                        Grad-CAM++ maps
  weights/model_seed{42,123,2024,3407,777}.pth
                                        the five final checkpoints (~133 MB
                                        each; host with Git LFS or Zenodo)

fusion/
  fusion_model.py                       5:3:2 weighted soft voting + post-hoc
                                        temperature scaling + bootstrap /
                                        weight-sensitivity / ablation analyses
  _fusion_common.py                     shared temperature-scaling module
                                        (single source of truth)
  models/fusion_temperature.json        final temperature T = 0.45 (fitted on
                                        Train, criterion RPS)
```

## Environment

Two interpreters were used (see versions actually used for the reported
results):

| Component | Python | Key packages |
|---|---|---|
| Clinical / habitat model / fusion | 3.9 | scikit-learn 1.6.1, PyRadiomics v3.1.0, SimpleITK 2.5.6, pandas, numpy, scipy, matplotlib |
| Deep learning | 3.10 (CUDA) | torch 2.8.0+cu128, torchvision, SimpleITK 2.5.4, nibabel 5.4.2, scikit-learn 1.7.2 |

```
pip install scikit-learn==1.6.1 pyradiomics==3.1.0 SimpleITK==2.5.6 \
            numpy pandas scipy matplotlib
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Input data expected by the code

* `CT/{ID}.nii.gz`, `mask/{ID}.nii.gz` — native CT (HU) and tumour mask,
  same grid.
* `00_Split/00_dataset_assignments.csv`, `00_clinical_all.csv`,
  `00_nodule_type.csv` — pre-defined cohort assignment and clinical table
  (study data; adjust the `*_DIR`/`*_SPLIT` constants at the top of each
  script to your paths).
* Clinical variables used by the final model (7):
  `Tumor_Markers, SCG, Morphology, Air_Bronchogram, Pleural_Tag, LD_mm, CTmean`.

## Reproducing the reported models

1. **Clinical** — `python clinical/clinical_model.py`. All randomness is
   seeded (`random_state=42`); the pipeline re-selects the same 7 features
   and the same RF hyperparameters as in
   `clinical/models/final_hyperparameters.txt`.
2. **Habitat** — run `habitat/habitat_feature_extraction.py` (Phase A
   refits the k-means centres deterministically; the shipped
   `models/pop_kmeans_centers.npz` is the one used in the paper), then
   `habitat/habitat_model_14features.py`, which reproduces the 14-feature
   signature and its per-class LASSO coefficients
   (`models/final_14feature_lasso_coefficients.csv`).
3. **Deep learning** — `deep_learning/preprocess_64cube.py` builds the
   64³ cache; `deep_learning/train_r3d18.py` defines and trains the model.
   The deployed ensemble uses the five shipped checkpoints
   (`deep_learning/weights/`); per-case probabilities are the mean softmax
   over the five seeds with 8-flip test-time augmentation
   (`inference_reference_gradcampp.py` shows exactly how the checkpoints are
   loaded and applied).
4. **Fusion** — `fusion/fusion_model.py` combines the three components'
   class probabilities with weights 0.5 / 0.3 / 0.2 and applies the shipped
   temperature `T = 0.45` (`models/fusion_temperature.json`, criterion RPS,
   fitted on Train). Temperature scaling is monotone per sample: accuracy,
   F1 and confusion matrices are unchanged; only probability-dependent
   metrics (one-vs-rest AUROC slightly, RPS and calibration) are affected.

## Notes

* The five `.pth` checkpoints total ≈ 650 MB. For GitHub, push them via
  **Git LFS**, or deposit them in a Zenodo repository and link it here.
* Patient-level data (imaging + clinical tables) are available from the
  corresponding author under reasonable request, per the data-availability
  statement of the manuscript.
