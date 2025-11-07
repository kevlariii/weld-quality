### Data Loader

from pathlib import Path
import pandas as pd

TARGET_COLS = ['yield_strength_MPa',
 'uts_MPa',
 'elongation_pct',
 'reduction_area_pct',
 'charpy_temp_C',
 'charpy_toughness_J']


def load_data(target: str, data_version: str = "pca", base_path: str = None):
    """
    Load pre-processed data from local files for semi-supervised learning.
    
    Parameters:
    -----------
    target : str
        Target variable name. Must be one of TARGET_COLS.
    data_version : str, default="pca"
        Version of data to load:
        - "pca": Load PCA-reduced features (X_train_pca.csv, etc.)
        - "clean": Load cleaned features (X_train_clean.csv, etc.)
    base_path : str, optional
        Base path to data directory. If None, uses default "../../data/data_splits"
        
    Returns:
    --------
    tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        Training, validation, and test sets ready for semi-supervised learning.
        
    Example:
    --------
    >>> # Load PCA-reduced data (default)
    >>> train, val, test = load_local_data("yield_strength_MPa")
    >>> X_train, y_train = train
    >>> X_val, y_val = val
    >>> X_test, y_test = test
    >>> 
    >>> # Load clean data instead
    >>> train, val, test = load_local_data("yield_strength_MPa", data_version="clean")
    """
    assert target in TARGET_COLS, f"Target must be one of {TARGET_COLS}"
    assert data_version in ["pca", "clean"], "data_version must be 'pca' or 'clean'"
    
    try:
        # Set up paths
        if base_path is None:
            # Get project root directory (where this file is at src/semi_supervised/loader.py)
            # Handle both script and notebook contexts
            try:
                project_root = Path(__file__).parent.parent.parent
            except NameError:
                # __file__ not defined in notebooks, use current working directory
                project_root = Path.cwd()
                if 'src' in str(project_root):
                    # If we're in a subdirectory, go up to project root
                    while project_root.name != 'weld-quality' and project_root.parent != project_root:
                        project_root = project_root.parent
            base_path = project_root / "data" / "data_splits"
        else:
            base_path = Path(base_path)
        
        target_dir = base_path / target
        
        # Check if directory exists
        if not target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {target_dir}")
        
        print(f"Loading {data_version.upper()} data for target: {target}")
        print(f"Data directory: {target_dir}")
        
        # Determine file suffix
        suffix = "_pca" if data_version == "pca" else "_clean"
        
        # Load features
        X_train = pd.read_csv(target_dir / f"X_train{suffix}.csv")
        X_val = pd.read_csv(target_dir / f"X_val{suffix}.csv")
        X_test = pd.read_csv(target_dir / f"X_test{suffix}.csv")
        
        # Load targets
        y_train = pd.read_csv(target_dir / "y_train.csv").squeeze()
        y_val = pd.read_csv(target_dir / "y_val.csv").squeeze()
        y_test = pd.read_csv(target_dir / "y_test.csv").squeeze()
        
        print(f"==> Data loaded successfully:")
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
        print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
        print(f"  Features: {X_train.shape[1]}")
        
        # Check for missing values
        missing_train = X_train.isnull().sum().sum()
        missing_val = X_val.isnull().sum().sum()
        missing_test = X_test.isnull().sum().sum()
        
        if missing_train + missing_val + missing_test > 0:
            print(f"/!\  Warning: Missing values detected!")
            print(f"   Train: {missing_train}, Val: {missing_val}, Test: {missing_test}")
        else:
            print(f"--> No missing values detected")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        
    except FileNotFoundError as e:
        print(f"XXX Error: {e}")
        if data_version == "pca":
            print(f"   Hint: PCA files not found. Run '04. pca.ipynb' first or use data_version='clean'")
        return None, None, None
    except Exception as e:
        print(f"XXX Error loading data: {e}")
        return None, None, None

