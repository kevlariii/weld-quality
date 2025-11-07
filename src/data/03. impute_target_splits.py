"""
Impute missing values in features for each target's data splits.

This script processes each target directory created in 02. data_preparation.ipynb:
- Loads X_train, X_val, X_test for each target
- Applies consistent imputation strategy across all splits
- Saves cleaned versions back to the same directories
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer

# Configuration
SPLITS_DIR = Path("../../data/data_splits")
DROP_THRESHOLD = 0.70  # Drop columns with >70% missing values

# Target columns (from 02. data_preparation.ipynb)
TARGET_COLS = [
    "yield_strength_MPa",
    "uts_MPa",
    "elongation_pct",
    "reduction_area_pct",
    "charpy_temp_C",
    "charpy_toughness_J"
]

# Composition columns to impute with mean
COMPOSITION_MEAN_COLS = ["sulfur_wt_pct", "phosphorus_wt_pct"]

# Composition columns to impute with 0 (not deliberately added)
COMPOSITION_ZERO_COLS = [
    'carbon_wt_pct', 'silicon_wt_pct', 'manganese_wt_pct', 
    'nickel_wt_pct', 'chromium_wt_pct', 'molybdenum_wt_pct', 
    'vanadium_wt_pct', 'copper_wt_pct', 'oxygen_ppm',
    'titanium_ppm', 'nitrogen_ppm', 'aluminium_ppm'
]


def impute_features_for_target(target_name):
    """
    Impute missing values in features for a specific target's splits.
    
    Strategy:
    1. Drop columns with >70% missing values (based on training set)
    2. Impute composition columns with mean or 0
    3. Impute remaining columns with most frequent value
    4. Fit imputers on training data only
    5. Apply to val and test sets
    """
    print(f"\n{'='*80}")
    print(f"Processing target: {target_name}")
    print(f"{'='*80}")
    
    target_dir = SPLITS_DIR / target_name
    
    # Check if directory exists
    if not target_dir.exists():
        print(f"/!\ Directory not found: {target_dir}")
        return
    
    # Load data
    print("\n[1/6] Loading data...")
    X_train = pd.read_csv(target_dir / "X_train.csv")
    X_val = pd.read_csv(target_dir / "X_val.csv")
    X_test = pd.read_csv(target_dir / "X_test.csv")
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    # Step 1: Drop sparse columns
    print(f"\n[2/6] Dropping columns with >{DROP_THRESHOLD*100:.0f}% missing values...")
    missing_pct = (X_train.isnull().sum() / len(X_train))
    cols_to_drop = missing_pct[missing_pct > DROP_THRESHOLD].index.tolist()
    
    if cols_to_drop:
        print(f"  Dropping {len(cols_to_drop)} columns:")
        for col in cols_to_drop:
            print(f"    - {col} ({missing_pct[col]*100:.1f}% missing)")
        
        X_train = X_train.drop(columns=cols_to_drop)
        X_val = X_val.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
    else:
        print(f"  --> No columns to drop")
    
    # Step 2: Impute composition columns with mean
    print(f"\n[3/6] Imputing composition columns (mean strategy)...")
    for col in COMPOSITION_MEAN_COLS:
        if col in X_train.columns:
            train_mean = X_train[col].mean()
            X_train[col].fillna(train_mean, inplace=True)
            X_val[col].fillna(train_mean, inplace=True)
            X_test[col].fillna(train_mean, inplace=True)
            print(f"  --> {col}: filled with mean ({train_mean:.4f})")
    
    # Step 3: Impute composition columns with 0
    print(f"\n[4/6] Imputing composition columns (zero strategy)...")
    for col in COMPOSITION_ZERO_COLS:
        if col in X_train.columns:
            X_train[col].fillna(0, inplace=True)
            X_val[col].fillna(0, inplace=True)
            X_test[col].fillna(0, inplace=True)
            print(f"  --> {col}: filled with 0")
    
    # Step 4: Impute remaining missing columns with most frequent
    print(f"\n[5/6] Imputing remaining columns (most frequent strategy)...")
    remaining_missing_cols = X_train.columns[X_train.isnull().any()].tolist()
    
    if remaining_missing_cols:
        print(f"  Imputing {len(remaining_missing_cols)} remaining columns:")
        for col in remaining_missing_cols:
            missing_count = X_train[col].isnull().sum()
            print(f"    - {col} ({missing_count} missing values)")
        
        # Fit imputer on training data
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(X_train[remaining_missing_cols])
        
        # Transform all sets
        X_train[remaining_missing_cols] = imputer.transform(X_train[remaining_missing_cols])
        X_val[remaining_missing_cols] = imputer.transform(X_val[remaining_missing_cols])
        X_test[remaining_missing_cols] = imputer.transform(X_test[remaining_missing_cols])
        
        print(f"\n  Most frequent values used:")
        for i, col in enumerate(remaining_missing_cols):
            print(f"    {col}: {imputer.statistics_[i]}")
    else:
        print(f"  --> No remaining columns with missing values")
    
    # Verification
    print(f"\n[6/6] Verification...")
    train_missing = X_train.isnull().sum().sum()
    val_missing = X_val.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    
    print(f"  Remaining missing values:")
    print(f"    Train: {train_missing}")
    print(f"    Val:   {val_missing}")
    print(f"    Test:  {test_missing}")
    
    if train_missing + val_missing + test_missing == 0:
        print(f"  ==> All missing values imputed successfully!")
    else:
        print(f"  /!\ Warning: {train_missing + val_missing + test_missing} missing values remain")
    
    # Save cleaned data
    print(f"\n  Saving cleaned datasets...")
    X_train.to_csv(target_dir / "X_train_clean.csv", index=False)
    X_val.to_csv(target_dir / "X_val_clean.csv", index=False)
    X_test.to_csv(target_dir / "X_test_clean.csv", index=False)
    
    print(f"  ==> Saved to: {target_dir}/")
    print(f"    Final shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    return {
        'target': target_name,
        'train_shape': X_train.shape,
        'val_shape': X_val.shape,
        'test_shape': X_test.shape,
        'cols_dropped': len(cols_to_drop),
        'success': (train_missing + val_missing + test_missing) == 0
    }


def main():
    """Process all targets."""
    print("=" * 80)
    print("IMPUTING FEATURES FOR ALL TARGET SPLITS")
    print("=" * 80)
    print(f"\nSplits directory: {SPLITS_DIR}")
    print(f"Targets to process: {len(TARGET_COLS)}")
    print(f"Drop threshold: {DROP_THRESHOLD*100:.0f}%")
    
    results = []
    
    for target in TARGET_COLS:
        try:
            result = impute_features_for_target(target)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nXXX Error processing {target}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nProcessed {len(results)}/{len(TARGET_COLS)} targets successfully\n")
    
    for result in results:
        status = "[SUCCESS]" if result['success'] else "[FAILURE]"
        print(f"{status} {result['target']}")
        print(f"   Train: {result['train_shape']}, Val: {result['val_shape']}, Test: {result['test_shape']}")
        print(f"   Dropped {result['cols_dropped']} sparse columns")
    
    print(f"\n{'='*80}")
    print("===> ALL TARGETS PROCESSED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
