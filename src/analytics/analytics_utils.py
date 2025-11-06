import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error

# 1. Analyze missing values in each column
def analyze_missing_values(df, threshold=0.8):
    """
    Analyze missing values and categorize columns based on missing percentage
    """
    missing_info = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_percentage': df.isnull().sum() / len(df),
        'data_type': df.dtypes
    })
    
    # Categorize columns
    missing_info['action'] = missing_info['missing_percentage'].apply(
        lambda x: 'drop' if x > threshold else 
                 ('impute_carefully' if x > 0.3 else 
                 ('impute_standard' if x > 0 and x <= 0.3 else 'keep'))
    )
    
    return missing_info.sort_values('missing_percentage', ascending=False)


def remove_high_missing_columns_keep_target(df, target:str, threshold=0.8):
    """Remove columns with missing values above threshold, except the target column"""
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[(missing_pct > threshold) & (missing_pct.index != target)].index.tolist()

    if cols_to_drop:
        print(f"Dropping columns with >{threshold*100}% missing values (excluding target '{target}'): {cols_to_drop}")
        df_cleaned = df.drop(columns=cols_to_drop)
    else:
        print("No columns exceed the missing value threshold (excluding target)")
        df_cleaned = df.copy()
    
    return df_cleaned, cols_to_drop

def remove_high_missing_columns(df, threshold=0.8):
    """Remove columns with missing values above threshold"""
    # missing_pct = df.isnull().sum() / len(df)
    # cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    missing_analysis = analyze_missing_values(df, threshold)
    cols_to_drop = missing_analysis[missing_analysis['action'] == 'drop']['column'].tolist()

    if cols_to_drop:
        print(f"Dropping columns with >{threshold*100}% missing values: {cols_to_drop}")
        df_cleaned = df.drop(columns=cols_to_drop)
    else:
        print("No columns exceed the missing value threshold")
        df_cleaned = df.copy()
    
    return df_cleaned, cols_to_drop


# 3. Choose imputation strategy based on missing percentage and data characteristics
def choose_imputation_strategy(df, column, missing_pct):
    """
    Choose the best imputation strategy based on:
    - Missing percentage
    - Data type
    - Distribution characteristics
    """
    col_data = df[column].dropna()
    
    # For categorical data
    if df[column].dtype == 'object' or df[column].dtype.name == 'category':
        if missing_pct < 0.1:
            return 'mode'
        elif missing_pct < 0.3:
            return 'mode_or_new_category'
        else:
            return 'new_category'
    
    # For numerical data
    else:
        # Check distribution
        skewness = col_data.skew()
        
        if missing_pct < 0.05:
            # Low missing: simple strategies work well
            return 'mean' if abs(skewness) < 1 else 'median'
        elif missing_pct < 0.15:
            # Medium missing: more sophisticated methods
            return 'knn' if len(col_data) > 100 else 'median'
        elif missing_pct < 0.3:
            # High missing: advanced methods or domain knowledge
            return 'iterative' if len(df.columns) > 5 else 'median'
        else:
            # Very high missing: consider dropping or domain-specific imputation
            return 'drop_or_domain_specific'

# 4. Apply different imputation strategies
# def apply_imputation(df, strategy_dict):
#     """Apply different imputation strategies to different columns"""
#     df_imputed = df.copy()
    
#     for column, strategy in strategy_dict.items():
#         if column not in df_imputed.columns:
#             continue
            
#         missing_mask = df_imputed[column].isnull()
        
#         if not missing_mask.any():
#             continue
            
#         if strategy == 'mean':
#             imputer = SimpleImputer(strategy='mean')
#             df_imputed[column] = imputer.fit_transform(df_imputed[[column]]).ravel()
            
#         elif strategy == 'median':
#             imputer = SimpleImputer(strategy='median')
#             df_imputed[column] = imputer.fit_transform(df_imputed[[column]]).ravel()
            
#         elif strategy == 'mode':
#             imputer = SimpleImputer(strategy='most_frequent')
#             df_imputed[column] = imputer.fit_transform(df_imputed[[column]]).ravel()
            
#         elif strategy == 'knn':
#             # Use only numeric columns for KNN
#             numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
#             knn_imputer = KNNImputer(n_neighbors=5)
#             df_numeric = df_imputed[numeric_cols]
#             df_imputed[numeric_cols] = knn_imputer.fit_transform(df_numeric)
            
#         elif strategy == 'iterative':
#             # Use only numeric columns for iterative imputation
#             numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
#             iterative_imputer = IterativeImputer(random_state=42, max_iter=10)
#             df_numeric = df_imputed[numeric_cols]
#             df_imputed[numeric_cols] = iterative_imputer.fit_transform(df_numeric)
            
#         elif strategy == 'new_category':
#             df_imputed[column] = df_imputed[column].fillna('Missing')
    
#     return df_imputed

def apply_imputation(df, strategy_dict):
    """Apply different imputation strategies to different columns"""
    df_imputed = df.copy()
    
    # Separate strategies that need to be applied column-wise vs all at once
    individual_strategies = {}
    batch_strategies = {'knn': [], 'iterative': []}
    
    for column, strategy in strategy_dict.items():
        if column not in df_imputed.columns:
            continue
            
        missing_mask = df_imputed[column].isnull()
        
        if not missing_mask.any():
            continue
            
        if strategy in ['knn', 'iterative']:
            batch_strategies[strategy].append(column)
        else:
            individual_strategies[column] = strategy
    
    # Apply individual strategies first
    for column, strategy in individual_strategies.items():
        missing_mask = df_imputed[column].isnull()
        
        if strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
            df_imputed[column] = imputer.fit_transform(df_imputed[[column]]).ravel()
            
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
            df_imputed[column] = imputer.fit_transform(df_imputed[[column]]).ravel()
            
        elif strategy == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[column] = imputer.fit_transform(df_imputed[[column]]).ravel()
            
        elif strategy == 'new_category':
            df_imputed[column] = df_imputed[column].fillna('Missing')
    
    # Apply KNN imputation if needed
    if batch_strategies['knn']:
        try:
            # Get all numeric columns that exist in the dataframe
            numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
            # Only use columns that actually exist and have data
            valid_numeric_cols = [col for col in numeric_cols if col in df_imputed.columns and not df_imputed[col].isnull().all()]
            
            if len(valid_numeric_cols) > 1:  # Need at least 2 columns for KNN
                knn_imputer = KNNImputer(n_neighbors=min(5, len(df_imputed)-1))
                df_numeric = df_imputed[valid_numeric_cols].copy()
                imputed_values = knn_imputer.fit_transform(df_numeric)
                df_imputed[valid_numeric_cols] = imputed_values
            else:
                # Fallback to median for KNN columns if not enough numeric columns
                for col in batch_strategies['knn']:
                    if col in df_imputed.columns:
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        except Exception as e:
            print(f"KNN imputation failed, falling back to median: {e}")
            for col in batch_strategies['knn']:
                if col in df_imputed.columns:
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
    
    # Apply iterative imputation if needed
    if batch_strategies['iterative']:
        try:
            # Get all numeric columns that exist in the dataframe
            numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
            # Only use columns that actually exist and have data
            valid_numeric_cols = [col for col in numeric_cols if col in df_imputed.columns and not df_imputed[col].isnull().all()]
            
            if len(valid_numeric_cols) > 1:  # Need at least 2 columns for iterative
                iterative_imputer = IterativeImputer(random_state=42, max_iter=10)
                df_numeric = df_imputed[valid_numeric_cols].copy()
                imputed_values = iterative_imputer.fit_transform(df_numeric)
                df_imputed[valid_numeric_cols] = imputed_values
            else:
                # Fallback to median for iterative columns if not enough numeric columns
                for col in batch_strategies['iterative']:
                    if col in df_imputed.columns:
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        except Exception as e:
            print(f"Iterative imputation failed, falling back to median: {e}")
            for col in batch_strategies['iterative']:
                if col in df_imputed.columns:
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
    
    return df_imputed




# 5. Validate imputation quality
def validate_imputation(original_df, imputed_df, sample_columns=None):
    """
    Validate imputation by comparing distributions and using cross-validation
    """
    if sample_columns is None:
        sample_columns = original_df.select_dtypes(include=[np.number]).columns[:5]
    
    fig, axes = plt.subplots(len(sample_columns), 2, figsize=(12, 3*len(sample_columns)))
    
    for i, col in enumerate(sample_columns):
        if col not in original_df.columns or col not in imputed_df.columns:
            continue
            
        # Original distribution (without missing values)
        original_data = original_df[col].dropna()
        imputed_data = imputed_df[col]
        
        axes[i, 0].hist(original_data, alpha=0.7, label='Original', bins=30)
        axes[i, 0].hist(imputed_data, alpha=0.7, label='After Imputation', bins=30)
        axes[i, 0].set_title(f'{col} - Distribution Comparison')
        axes[i, 0].legend()
        
        # Missing value pattern
        axes[i, 1].scatter(range(len(original_df)), original_df[col], alpha=0.6, s=1, label='Original')
        axes[i, 1].scatter(range(len(imputed_df)), imputed_df[col], alpha=0.6, s=1, label='Imputed')
        axes[i, 1].set_title(f'{col} - Missing Value Pattern')
        axes[i, 1].legend()
    
    plt.tight_layout()
    plt.show()


