### Un fichier pour gérer le chargement, l'imputation et le split des données en semi-supervisé



from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from analytics.analytics_utils import (
    remove_high_missing_columns_keep_target,
    choose_imputation_strategy,
    apply_imputation
)

TARGET_COLS = ['yield_strength_MPa',
 'uts_MPa',
 'elongation_pct',
 'reduction_area_pct',
 'charpy_temp_C',
 'charpy_toughness_J']

POSSIBLE_SPLITS = ['charpy_temp_C_labeled',
 'charpy_temp_C_unlabeled',
 'charpy_toughness_J_labeled',
 'charpy_toughness_J_unlabeled',
 'elongation_pct_labeled',
 'elongation_pct_unlabeled',
 'reduction_area_pct_labeled',
 'reduction_area_pct_unlabeled',
 'uts_MPa_labeled',
 'uts_MPa_unlabeled',
 'yield_strength_MPa_labeled',
 'yield_strength_MPa_unlabeled']

DATASET_NAME = "MoSBAIHI/weld-quality-dataset"

def name_column(col_name, labeled=True):
    if labeled:
        return f"{col_name}_labeled"
    else:
        return f"{col_name}_unlabeled"

def load_data_with_token(target:str, split:str, HF_API_KEY=None):
    assert target in TARGET_COLS, f"Target must be one of {TARGET_COLS}"
    assert split in ["labeled", "unlabeled"], "Split must be 'labeled' or 'unlabeled'"
    try:
        if HF_API_KEY:
            print("Loading dataset from Hugging Face Hub...")
            split_name = name_column(target, labeled=(split == "labeled"))
            dataset = load_dataset(DATASET_NAME, split=split_name, token=HF_API_KEY)
            todrop_cols = [col for col in TARGET_COLS if col != target]
            dataset = dataset.remove_columns(todrop_cols)

            print("Imputing missing values...")
            df = dataset.to_pandas()
            df_clean, _ = remove_high_missing_columns_keep_target(df, target, threshold=0.8)
            imputation_strategies = {}
            for col in df_clean.columns:
                if col != target:
                    missing_pct = df_clean[col].isnull().sum() / len(df_clean)
                    if missing_pct > 0:
                        strategy = choose_imputation_strategy(df_clean, col, missing_pct)
                        imputation_strategies[col] = strategy
            df_imputed = apply_imputation(df_clean, imputation_strategies)
            print("Data loading and imputation completed.")
            return df_imputed
        else:
            raise ValueError("HF_API_KEY is required to load the dataset from Hugging Face Hub.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    

def load_data(target:str, split:str):
    assert target in TARGET_COLS, f"Target must be one of {TARGET_COLS}"
    assert split in ["labeled", "unlabeled"], "Split must be 'labeled' or 'unlabeled'"
    try:
        print("Loading dataset from Hugging Face Hub...")
        split_name = name_column(target, labeled=(split == "labeled"))
        dataset = load_dataset(DATASET_NAME, split=split_name)
        todrop_cols = [col for col in TARGET_COLS if col != target]
        dataset = dataset.remove_columns(todrop_cols)

        print("Imputing missing values...")
        df = dataset.to_pandas()
        df_clean, _ = remove_high_missing_columns_keep_target(df, target, threshold=0.8)
        # print(f"Columns after cleaning: {df_clean.columns}")
        imputation_strategies = {}
        for col in df_clean.columns:
            if col != target:
                missing_pct = df_clean[col].isnull().sum() / len(df_clean)
                if missing_pct > 0:
                    strategy = choose_imputation_strategy(df_clean, col, missing_pct)
                    imputation_strategies[col] = strategy
        
        df_imputed = apply_imputation(df_clean, imputation_strategies)
        # print(f"Columns after imputation: {df_imputed.columns}")
        print("Data loading and imputation completed.")
        return df_imputed
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    
def split_train_val_test(df_labeled, df_unlabeled, target:str, val_size=0.2, test_size=0.1, random_state=42):
    assert target in TARGET_COLS, f"Target must be one of {TARGET_COLS}"
    try: 
        print("Splitting data into train, validation, and test sets...")
        X_labeled = df_labeled.drop(columns=[target], axis=1)
        y_labeled = df_labeled[target]
        X_unlabeled = df_unlabeled.drop(columns=[target], axis=1)
        y_unlabeled = df_unlabeled[target]

        # split labeled data into train, val, test
        X_train, X_temp, y_train, y_temp = train_test_split(X_labeled, y_labeled, test_size=(val_size + test_size), random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (val_size + test_size)), random_state=random_state)

        # Combine unlabeled data with training data
        X_train_final = pd.concat([X_train, X_unlabeled], ignore_index=True)
        y_train_final = pd.concat([y_train, y_unlabeled], ignore_index=True)

        return (X_train_final, y_train_final), (X_val, y_val), (X_test, y_test)
    except Exception as e:
        print(f"Error during train/val/test split: {e}")
        return None, None, None

