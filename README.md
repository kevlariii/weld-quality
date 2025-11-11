# Weld Quality — Semi-Supervised Learning Pipeline

Comprehensive project for predicting weld quality using semi-supervised learning (SSL). The repository provides data preparation, exploratory analysis, optional PCA-based dimensionality reduction, and an SSL experimentation pipeline that converts continuous material properties into three quality classes (Bad / Medium / Good) and evaluates both SSL-only and RF-on-SSL approaches.

This README gives a clear quick-start path, explains the project structure in detail, documents how to use each component, and shows how to reproduce experiments.

## Table of Contents
- [Project Overview](#project-overview)
- [Quick Start](#quick-start-5-minutes)
- [Installation](#installation)
- [Repository Structure (Detailed)](#repository-structure-detailed)
- [Data Pipeline & Notebooks](#data-pipeline--notebooks)
- [Semi-Supervised Pipeline (How to Run)](#semi-supervised-pipeline-how-to-run)
- [Outputs and File Layout](#outputs-and-file-layout)
- [Reproducibility and Best Practices](#reproducibility-and-best-practices)

## Project Overview

Motivation: weld testing yields continuous properties (strength, toughness, elongation). Labels are expensive, so this project demonstrates how to leverage unlabeled samples via semi-supervised methods (LabelPropagation, LabelSpreading, SelfTraining) and then train supervised models (RandomForest) on propagated labels.

**Key features:**
- Data-driven tertile-based discretization of continuous targets (avoids hard-coded thresholds)
- Optional PCA pipeline to reduce dimensionality while keeping ~90% variance
- Hyperparameter tuning utilities for SSL and RF (configurable and optional)
- Reproducible results: saved thresholds, tuning traces, plots, and comparison tables
- Clean separation of data preparation, EDA, and experimentation notebooks

## Quick Start (5 minutes)

1. **Create a Python environment and install dependencies:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Prepare data** (if not already present): run the data pipeline notebooks. See [Data Pipeline & Notebooks](#data-pipeline--notebooks) for details.

3. **Run the semi-supervised notebook:**

- Open `src/semi_supervised/semi_supervied_training.ipynb` in Jupyter Lab or Notebook
- Set `TARGET_NAME` (e.g., `yield_strength_MPa`) and `DATA_VERSION` (`pca` or `clean`)
- Set `ENABLE_TUNING` to `False` for a quick run, `True` for hyperparameter searches
- Run all cells sequentially

Alternatively, run individual data-prep notebooks first (EDA, PCA) to inspect or precompute artifacts.

## Installation

Install all dependencies:

```powershell
pip install -r requirements.txt
```

**Main packages used:**
- `pandas`, `numpy` : data manipulation
- `scikit-learn` : models, PCA, semi-supervised utilities
- `matplotlib`, `seaborn` : plotting and visualization
- `jupyter` or `jupyterlab` : interactive notebook execution

## Repository Structure (Detailed)

This section documents the contents and purpose of each folder and file so you know where to look and how to reuse components.

**Root files:**
- `README.md` : This file. Quick-start, detailed usage, and directory map
- `requirements.txt` : Python dependencies. Install with `pip install -r requirements.txt`

**`data/` — Main data folder:**
- `clean_weld_quality_dataset.csv` : the cleaned, canonical dataset used as input for split creation
- `data_splits/<target>/` : per-target subfolders (one per target variable) containing CSV splits
  - `X_train.csv`, `y_train.csv` : raw training split (may include unlabeled rows in `y_train`)
  - `X_train_clean.csv`, `X_val_clean.csv`, `X_test_clean.csv` : imputed/clean feature matrices (prepared by `03.impute_target_splits.py`)
  - `X_train_pca.csv`, `X_val_pca.csv`, `X_test_pca.csv` : PCA-reduced features (optional — produced by `src/data/05. pca.ipynb`)

**`figs/` — Saved figures (organized by type):**
- `figs/classes/` : class distribution and threshold visualization plots
- `figs/pca/` : PCA projection plots
- `figs/all_features/` : per-feature distribution and EDA visuals

**`results/` — Experiment outputs and summaries:**
- `results/ssl_results/<target>/(pca|all_features)/` : predictions, tuning CSVs, thresholds, model comparisons per run
- `results/_comparisons/` : aggregated comparisons across targets or runs

**`src/` — Source code and notebooks:**
- `src/data/` : data cleaning, preparation, imputation, EDA, and PCA
  - `01.dataset_cleaning.ipynb` : raw data cleaning
  - `02. data_preparation.ipynb` : creates semi-supervised splits per target
  - `03. impute_target_splits.py` : imputes missing values, writes `X_*_clean.csv`
  - `04. eda.ipynb` : target-specific exploratory data analysis
  - `05. pca.ipynb` : PCA pipeline, writes `X_*_pca.csv`
- `src/semi_supervised/` : SSL experiments and helpers
  - `semi_supervied_training.ipynb` :**main SSL experiment notebook** (configure, run, save results)
  - `loader.py` : helper to load PCA or clean splits programmatically
  - `ssl_training.py` : functions for discretization and SSL/RF tuning
- `src/utils/` : utility functions and helpers
- `src/compare/` : notebooks and scripts to aggregate results across targets

**Where to look first:**
- To **run experiments now**: open `src/semi_supervised/semi_supervied_training.ipynb`
- To **prepare data**: `src/data/01.dataset_cleaning.ipynb` → `02. data_preparation.ipynb` → `03. impute_target_splits.py`
- For **optional PCA**: run `src/data/05. pca.ipynb` after imputation

**Programmatic usage examples:**

Load data from Python scripts or notebooks:

```python
from src.semi_supervised.loader import load_data

# Load PCA features for a target
(train, val, test) = load_data('yield_strength_MPa', data_version='pca')
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test
```

Use SSL helper functions:

```python
from src.semi_supervised.ssl_training import convert_to_quality_classes_tertiles

y_train_c, y_val_c, y_test_c, low, high = convert_to_quality_classes_tertiles(y_train, y_val, y_test)
# Pass y_train_c to SSL models or tune_ssl_model
```

## Project Structure

```
weld-quality/
├── README.md
├── requirements.txt
├── data/
│   ├── clean_weld_quality_dataset.csv # Cleaned raw dataset
│   └── data_splits/                   # Per-target train/val/test splits
│       ├── charpy_temp_C/
│       ├── charpy_toughness_J/
│       ├── elongation_pct/
│       ├── reduction_area_pct/
│       ├── uts_MPa/
│       └── yield_strength_MPa/
├── figs/                              # Visualization and analysis figures
│   ├── all_features/
│   ├── classes/
│   └── pca/
├── results/
│   ├── ssl_results/                   # Semi-supervised learning outputs
│   └── _comparisons/                  # Model comparison results
└── src/
    ├── compare/                       # Model comparison scripts
    ├── data/                          # Data cleaning and preprocessing
    ├── semi_supervised/               # Semi-supervised learning pipelines
    └── utils/                         # Helper functions and utilities
```

## Data Pipeline & Notebooks

Location: `data/` and `src/data/`

1. **Data cleaning** : Notebook: `src/data/01.dataset_cleaning.ipynb`
   - Output: `data/clean_weld_quality_dataset.csv`

2. **Create semi-supervised splits** : Notebook: `src/data/02. data_preparation.ipynb`
   - Creates per-target `data/data_splits/<target>/` subfolders with `X_*.csv` and `y_*.csv`
   - Splits: 60% labeled + 100% unlabeled in training; 20% in val/test

3. **Imputation** : Script: `src/data/03. impute_target_splits.py`
   - Fills missing values (mean for numeric, mode for categorical)
   - Output: `X_*_clean.csv` files per target

4. **Exploratory Data Analysis** : Notebook: `src/data/04. eda.ipynb`
   - Distribution plots, correlation heatmaps, feature-target scatter plots

5. **PCA (optional)** : Notebook: `src/data/05. pca.ipynb`
   - Reduces to ~90% variance; output: `X_*_pca.csv`

## Semi-Supervised Pipeline (How to Run)

**Main notebook:** `src/semi_supervised/semi_supervied_training.ipynb`

**Configuration** (first code cell):
- `TARGET_NAME` : one of: `yield_strength_MPa`, `uts_MPa`, `elongation_pct`, `reduction_area_pct`, `charpy_temp_C`, `charpy_toughness_J`
- `DATA_VERSION` : `pca` (use `X_*_pca.csv`) or `clean` (use `X_*_clean.csv`)
- `ENABLE_TUNING` : `True` to run hyperparameter searches (slow), `False` for defaults

**What the notebook does:**
1. Loads data splits using `src/semi_supervised/loader.py`
2. Converts continuous targets to 3 classes (tertiles: Bad/Medium/Good)
3. Trains SSL methods (LabelPropagation, LabelSpreading, SelfTraining)
4. Propagates labels to unlabeled training samples
5. Trains RandomForest on propagated labels
6. Evaluates all models and saves results under `results/ssl_results/<target>/`

## Outputs and File Layout

**Key output paths** (per target):
- `results/ssl_results/<target>/pca|all_features/` — predictions, tuning logs, summaries
- `figs/` — saved plots (class distributions, confusion matrices, feature importance)

**Important output files:**
- `model_comparison.csv` — validation/test metrics for all pipelines
- `quality_thresholds.csv` — tertile thresholds used for discretization
- `*_predictions.csv` — predictions per method
- `*_tuning_results.csv` — tuning traces (when `ENABLE_TUNING=True`)

## Reproducibility and Best Practices

- **Random seeds:** fixed `random_state=42` throughout
- **No data leakage:** all imputers and scalers fitted on training data only
- **Tertile computation:** computed from labeled training samples only
- **Scalability:** tune hyperparameters judiciously; large grids can be time-consuming
