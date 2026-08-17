# Predicting Depression Using Physical Activity, Sleep, and Demographic Factors in NHANES: A Fairness-Centred Machine Learning Approach

Code accompanying the paper presented at **EMBC 2026** (IEEE Engineering in Medicine and Biology Conference). The paper is not yet available on IEEE Xplore; a link will be added once published.

## Overview

This repository contains the analysis pipeline for predicting depression (PHQ-9 total score ≥ 10) from accelerometer-derived physical activity and sleep features plus demographic factors in NHANES 2013–2014, with an emphasis on fairness across demographic subgroups.

The pipeline covers three stages:

1. **Data processing** — merge raw NHANES accelerometer, demographic, and PHQ-9 files; derive per-participant activity and sleep features.
2. **Modelling** — VIF-based feature selection, distribution-driven scaling, and grid-searched model selection across three feature sets (accelerometer only, demographics only, combined), with and without class-imbalance sampling.
3. **Fairness and interpretability** — subgroup-stratified performance, fairness metrics, permutation importance, and SHAP-based feature attribution.

## Contents

| File | Description |
|---|---|
| `data_processing.ipynb` | Loads `PAXMIN_H.xpt`, `DEMO_H.xpt`, and `DPQ_H.xpt`; cleans and reduces the cohort; derives physical activity features (activity-intensity minutes, daily mean MIMS, peak-30 MIMS, time-of-day MIMS fractions) and sleep features (average sleep hours, sleep SD, wake bouts via `skdh`); writes the merged analysis table. |
| `final_models.ipynb` | Exploratory plots, VIF feature selection (threshold 10), engineered features (`circadian_ratio`, `composite_mims`, `high_v_low`, `Income_per_person`), per-feature scaler assignment (Shapiro/skew/outlier rules), stratified train–test split, and `GridSearchCV` over Logistic Regression, Random Forest, Gradient Boosting, SVC, KNN, XGBoost, and LightGBM — optimised for F1 with 5-fold stratified CV, with SMOTE and random undersampling variants. Best models and their metadata are written to `Results/`. |
| `fairness.ipynb` | Loads the saved models and evaluates subgroup-stratified F1 and fairness metrics across sex, age band, race, citizenship, education, marital status, household size, household income, and poverty ratio; permutation importance and SHAP feature attribution. |

## Requirements

Python 3.9+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels` (VIF), `xgboost`, `lightgbm`, `imbalanced-learn` (SMOTE / undersampling), `shap`, and `scikit-digital-health` (`skdh`, used for wake-bout counts).

## Data

Data are from the [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/index.htm), **2013–2014 cycle**, and are not redistributed here. Download the following from the [NHANES 2013–2014 data overview](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/overview.aspx?BeginYear=2013) and place them in `Data/`:

- `PAXMIN_H.xpt` — minute-level physical activity monitor data
- `DEMO_H.xpt` — demographic variables
- `DPQ_H.xpt` — mental health / PHQ-9 depression screener

`Data/` and `Results/` are tracked as empty folders (`.gitkeep`) and their contents are excluded via `.gitignore`, since the raw accelerometer file and intermediate artefacts are too large to version.

## Usage

Run the notebooks in order:

```
data_processing.ipynb  →  final_models.ipynb  →  fairness.ipynb
```

1. **`data_processing.ipynb`** reads the three `.xpt` files from `Data/` and writes intermediates (`accelerometer_data_start.pkl`, `accelerometer_data.pkl`) plus the final merged table `demographics_data.csv` to `Data/`.
2. **`final_models.ipynb`** trains and selects models, writing to `Results/`:
   - `accel_best_model.pkl` / `accel_best_model_info.json`
   - `demo_best_model.pkl` / `demo_best_model_info.json`
   - `combined_best_model.pkl` / `combined_best_model_info.json`
   - the corresponding `*_sampling.pkl` / `*_sampling_info.json` variants
3. **`fairness.ipynb`** loads those models and produces the subgroup performance, fairness, and interpretability results.

All paths are relative to the repository root, so run the notebooks with the repository root as the working directory.

## Reproducibility

Splits and model fitting use `random_state=42`; the train–test split is stratified on the depression label. Exact numbers may still vary slightly with package versions, so pin versions (`pip freeze > requirements.txt`) if you need bit-for-bit reproduction.

## Citation

If you use this code, please cite:

> _[Full citation to be added once the paper is published on IEEE Xplore]_

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
