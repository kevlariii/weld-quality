# Weld Quality Prediction - Semi-Supervised Learning Pipeline

A machine learning pipeline for predicting weld quality using semi-supervised learning techniques. The project converts continuous weld properties into quality classes (Bad, Medium, Good) and leverages both labeled and unlabeled data for improved predictions.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [Usage](#usage)
- [Semi-Supervised Learning](#semi-supervised-learning)
- [Results](#results)

## Project Structure

```
weld-quality/
├── README.md
├── requirements.txt
├── config/
│   └── keys.json                      # API keys and configuration
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
│   ├── all_features/                  # Feature-based visualizations
│   ├── classes/                       # Class distribution plots
│   └── pca/                           # PCA component visualizations
├── results/
│   ├── ssl_results/                   # Semi-supervised learning outputs
│   │   ├── charpy_temp_C/
│   │   │   ├── all_features/
│   │   │   │   └── tuning/            # Hyperparameter tuning results
│   │   │   └── pca/
│   │   │       └── tuning/
│   │   ├── charpy_toughness_J/
│   │   │   ├── all_features/
│   │   │   │   └── tuning/
│   │   │   └── pca/
│   │   │       └── tuning/
│   │   ├── elongation_pct/
│   │   │   ├── all_features/
│   │   │   │   └── tuning/
│   │   │   └── pca/
│   │   │       └── tuning/
│   │   ├── reduction_area_pct/
│   │   │   ├── all_features/
│   │   │   │   └── tuning/
│   │   │   └── pca/
│   │   │       └── tuning/
│   │   ├── uts_MPa/
│   │   │   ├── all_features/
│   │   │   │   └── tuning/
│   │   │   └── pca/
│   │   │       └── tuning/
│   │   └── yield_strength_MPa/
│   │       ├── all_features/
│   │       │   └── tuning/
│   │       └── pca/
│   │           └── tuning/
│   └── _comparisons/                  # model comparison results
└── src/
    ├── analytics/                     # Analysis utilities and notebooks
    ├── compare/                       # Model comparison scripts
    ├── data/                          # Data cleaning and preprocessing
    ├── pre-processing/                # Feature engineering and scaling
    ├── semi_supervised/               # Semi-supervised learning pipelines
    │   └── __pycache__/
    └── utils/                         # Helper functions and shared utilities

```

## Installation

### 1. Install Dependencies
Use `uv` or `pip` to install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Required Packages
- pandas, numpy
- scikit-learn (semi-supervised learning, PCA, imputation)
- matplotlib, seaborn (visualization)
- jupyter/notebook (for running notebooks)

## Data Pipeline

### Step 1: Data Cleaning
**Notebook:** `src/data/01.dataset_cleaning.ipynb`

Loads and cleans the raw weld quality dataset from Parquet files.

```python
# Load from local file
df = pd.read_parquet("path/to/weld_data.parquet")
```

**Output:** `data/clean_weld_quality_dataset.csv`

### Step 2: Data Preparation (Semi-Supervised Splits)
**Notebook:** `src/data/02.data_preparation.ipynb`

Creates train/val/test splits for each target variable with a semi-supervised setup:
- **Training set:** 60% labeled + 100% unlabeled data
- **Validation set:** 20% labeled data
- **Test set:** 20% labeled data

**Targets:**
1. `yield_strength_MPa` - Yield strength
2. `uts_MPa` - Ultimate tensile strength
3. `elongation_pct` - Elongation percentage
4. `reduction_area_pct` - Reduction in area
5. `charpy_temp_C` - Charpy impact temperature
6. `charpy_toughness_J` - Charpy toughness

**Output:** `data/data_splits/{target}/X_train.csv, y_train.csv, ...`

### Step 3: Feature Imputation
**Script:** `src/data/03.impute_target_splits.py`

Automatically imputes missing values for all target datasets using:
- **Composition elements:** Mean imputation (S, P) or 0 (not-added elements)
- **Process parameters:** Mode imputation (categorical variables)

Run the script:
```bash
cd src/data
python 03.impute_target_splits.py
```

**Output:** `data/data_splits/{target}/X_train_clean.csv, X_val_clean.csv, X_test_clean.csv`

### Step 4: Exploratory Data Analysis (EDA)
**Notebook:** `src/data/04.eda.ipynb`

Performs target-specific EDA:
- Distribution plots
- Correlation heatmaps
- Feature-target scatter plots
- Outlier detection

**Usage:** Change `TARGET_NAME` variable to analyze different targets.

### Step 5: PCA Dimensionality Reduction (Optional)
**Notebook:** `src/data/04.pca.ipynb`

Reduces feature dimensionality using PCA while retaining 90% variance:
- Scales features using StandardScaler
- Fits PCA on training data
- Transforms all datasets consistently
- Visualizes variance and projections

**Output:** `data/data_splits/{target}/X_train_pca.csv, X_val_pca.csv, X_test_pca.csv`

## Usage

### Running the Semi-Supervised Learning Pipeline

**Notebook:** `src/semi_supervised/semi_supervied_training.ipynb`

#### 1. Configure the Pipeline
```python
TARGET_NAME = "yield_strength_MPa"  # Choose target
DATA_VERSION = "pca"  # Options: "pca" or "clean"
```

#### 2. The Pipeline Automatically:
1. **Loads data** from `data/data_splits/{target}/`
2. **Converts continuous values to quality classes** using tertiles:
   - **Bad (0):** Bottom third (< 33.3 percentile)
   - **Medium (1):** Middle third (33.3-66.6 percentile)
   - **Good (2):** Top third (> 66.6 percentile)
3. **Trains semi-supervised models:**
   - Label Propagation
   - Label Spreading
4. **Trains Random Forest classifiers** on propagated labels
5. **Evaluates all models** on validation and test sets
6. **Saves results** to `data/ssl_results/{target}/`

#### 3. Run All Cells
Simply execute all cells in order. The notebook will:
- Load your chosen data version (PCA or clean)
- Train 4 models (2 SSL + 2 RF)
- Display accuracy metrics
- Show confusion matrices
- Save predictions and comparisons

## Semi-Supervised Learning

### Why Semi-Supervised Learning?
The dataset contains both labeled and unlabeled samples. Semi-supervised learning leverages unlabeled data to improve model performance when labeled data is limited.

### Models Used

#### 1. Label Propagation
- Uses graph-based approach to propagate labels
- Hard clamping of labeled data
- RBF kernel with γ=20

#### 2. Label Spreading
- Soft clamping with α=0.2 (regularization)
- More robust to noise than Label Propagation
- RBF kernel with γ=20

#### 3. Random Forest (with SSL labels)
- Trained on SSL-propagated labels
- 100 trees, max_depth=10
- Uses all training data (labeled + newly labeled by SSL)

### Quality Classification Strategy
Instead of fixed thresholds, the pipeline uses **data-driven tertiles**:
- Computes 33.3% and 66.6% percentiles from labeled training data
- Automatically adapts to each target's distribution
- Ensures balanced class distribution (~33% per class)

## Results

Results are saved to `data/ssl_results/{target}/`:

### Files Generated
1. **Model Predictions:**
   - `LabelPropagation_predictions.csv`
   - `LabelSpreading_predictions.csv`
   - `RF_LabelPropagation_predictions.csv`
   - `RF_LabelSpreading_predictions.csv`

2. **Model Comparison:**
   - `model_comparison.csv` - Accuracy comparison table

3. **Quality Thresholds:**
   - `quality_thresholds.csv` - Tertile boundaries used

4. **Feature Importance (if using clean data):**
   - `feature_importance.csv` - RF feature rankings

### Expected Performance
- **Validation Accuracy:** 60-85% (varies by target)
- **Test Accuracy:** Similar to validation
- Random Forest typically outperforms pure SSL methods
- PCA features often perform similarly to clean features with fewer dimensions

## Complete Workflow

```bash
# 1. Clean raw data
# Run: src/data/01.dataset_cleaning.ipynb

# 2. Create semi-supervised splits
# Run: src/data/02.data_preparation.ipynb

# 3. Impute missing values
cd src/data
python 03.impute_target_splits.py

# 4. (Optional) Perform EDA
# Run: src/data/04.eda.ipynb

# 5. (Optional) Apply PCA
# Run: src/data/04.pca.ipynb

# 6. Train semi-supervised models
# Run: src/semi_supervised/semi_supervied_training.ipynb
```

## Notes

- **Data Leakage Prevention:** All imputers and scalers are fitted only on training data
- **Reproducibility:** Random state is set to 42 throughout the pipeline
- **Unlabeled Data:** Marked with NaN in target columns, converted to -1 for SSL
- **Flexible:** Easy to switch between PCA and clean features
- **Scalable:** Process all 6 targets by changing configuration
