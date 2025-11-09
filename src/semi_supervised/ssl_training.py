from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
import pandas as pd 
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression


def convert_to_quality_classes_tertiles(y_train, y_val, y_test):
    """
    Convert continuous target values to quality classes using tertiles.
    
    Thresholds are computed ONLY from labeled training data to prevent data leakage.
    
    Classes:
        0: Bad quality    (< 33.3 percentile)
        1: Medium quality (33.3 - 66.6 percentile)
        2: Good quality   (> 66.6 percentile)
       -1: Unlabeled      (for semi-supervised learning)
    
    Args:
        y_train, y_val, y_test: Target Series (may contain NaN for unlabeled)
    
    Returns:
        Classified Series for train/val/test, and low/high thresholds
    """
    # Get labeled training data only
    labeled_y_train = y_train.dropna()
    
    if len(labeled_y_train) == 0:
        raise ValueError("XXX No labeled training data available!")
    
    # Compute tertile thresholds
    threshold_low = labeled_y_train.quantile(1/3)
    threshold_high = labeled_y_train.quantile(2/3)
    
    print(f"Tertile Thresholds (from {len(labeled_y_train)} labeled samples):")
    print(f"  Bad/Medium:   {threshold_low:.2f}")
    print(f"  Medium/Good:  {threshold_high:.2f}")
    
    def classify(y):
        """Apply classification to any dataset"""
        y_classes = pd.Series(index=y.index, dtype=int)
        unlabeled_mask = y.isna()
        
        # Classify labeled data
        y_classes[~unlabeled_mask & (y < threshold_low)] = 0
        y_classes[~unlabeled_mask & (y >= threshold_low) & (y < threshold_high)] = 1
        y_classes[~unlabeled_mask & (y >= threshold_high)] = 2
        
        # Mark unlabeled with -1
        y_classes[unlabeled_mask] = -1
        
        return y_classes
    
    return (classify(y_train), classify(y_val), classify(y_test), 
            threshold_low, threshold_high)

# SSL hyperparameter tuning function

def _make_base_estimator(name_or_est):
    # Accept an already-built estimator or a short string
    if hasattr(name_or_est, "fit"):
        return name_or_est
    name = str(name_or_est).lower()
    if name in ("logreg", "logisticregression", "lr"):
        return LogisticRegression(max_iter=1000)
    if name in ("rf", "randomforest", "randomforestclassifier"):
        return RandomForestClassifier(n_estimators=200, random_state=42)
    raise ValueError(f"Unknown base_estimator '{name_or_est}'")

def tune_ssl_model(ModelClass, param_grid, X_train, y_train_classes, X_val, y_val_classes, method_name):
    print(f"\n{'='*70}")
    print(f"TUNING {method_name}")
    print(f"{'='*70}")

    best_score = 0.0
    best_params, best_model = None, None
    results = []

    def _eval(model, params):
        nonlocal best_score, best_params, best_model, results
        model.fit(X_train, y_train_classes)
        y_pred = model.predict(X_val)
        score = f1_score(y_val_classes, y_pred, average='weighted')
        results.append({**params, 'val_f1_score': score})
        if score > best_score:
            best_score, best_params, best_model = score, params, model

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='max_iter.*without convergence')

        # === Decisive routing: if no 'kernel' => SelfTraining ===
        if 'kernel' not in param_grid:
            base_estimators = param_grid.get('base_estimator', ['LogisticRegression'])
            thresholds     = param_grid.get('threshold', [0.8])
            max_iters      = param_grid.get('max_iter', [20])
            verboses       = param_grid.get('verbose', [False])

            for be in base_estimators:
                try:
                    base = _make_base_estimator(be)
                except Exception as e:
                    print(f"  [!]  Skipped base_estimator={be}: {str(e)[:80]}")
                    continue
                for thr in thresholds:
                    for mi in max_iters:
                        for vb in verboses:
                            params = {
                                'base_estimator': type(base).__name__,
                                'threshold': float(thr),
                                'max_iter': int(mi),
                                'verbose': bool(vb)
                            }
                            try:
                                model = SelfTrainingClassifier(
                                    base_estimator=base,
                                    threshold=float(thr),
                                    max_iter=int(mi),
                                    verbose=bool(vb)
                                )
                                _eval(model, params)
                            except Exception as e:
                                print(f"  [!]  Skipped {params}: {str(e)[:80]}")
        else:
            # === LabelPropagation / LabelSpreading ===
            kernels = param_grid.get('kernel', [])
            for kernel in kernels:
                if kernel == 'rbf':
                    for gamma in param_grid.get('gamma', []):
                        params = {'kernel': kernel, 'gamma': gamma, 'max_iter': param_grid.get('max_iter', [1000])[0]}
                        if 'alpha' in param_grid:  # LabelSpreading
                            for alpha in param_grid.get('alpha', []):
                                p = {**params, 'alpha': alpha}
                                try:
                                    model = ModelClass(**p, tol=1e-3)
                                    _eval(model, p)
                                except Exception as e:
                                    print(f"  [!]  Skipped {p}: {str(e)[:80]}")
                        else:  # LabelPropagation
                            try:
                                model = ModelClass(**params, tol=1e-3)
                                _eval(model, params)
                            except Exception as e:
                                print(f"  [!]  Skipped {params}: {str(e)[:80]}")

                elif kernel == 'knn':
                    for n_neighbors in param_grid.get('n_neighbors', []):
                        params = {'kernel': kernel, 'n_neighbors': n_neighbors, 'max_iter': param_grid.get('max_iter', [1000])[0]}
                        if 'alpha' in param_grid:  # LabelSpreading
                            for alpha in param_grid.get('alpha', []):
                                p = {**params, 'alpha': alpha}
                                try:
                                    model = ModelClass(**p, tol=1e-3)
                                    _eval(model, p)
                                except Exception as e:
                                    print(f"  [!]  Skipped {p}: {str(e)[:80]}")
                        else:  # LabelPropagation
                            try:
                                model = ModelClass(**params, tol=1e-3)
                                _eval(model, params)
                            except Exception as e:
                                print(f"  [!]  Skipped {params}: {str(e)[:80]}")

    if not results:
        print("\n===> Tested 0 parameter combinations (all failed).")
        return None, None, 0.0, pd.DataFrame(columns=['val_f1_score'])

    results_df = pd.DataFrame(results).sort_values('val_f1_score', ascending=False)
    print(f"\n===> Tested {len(results)} parameter combinations")
    print("\nTop 3 configurations:")
    print(results_df.head(3).to_string(index=False))
    print(f"\n{'─'*70}")
    print(f"BEST: {best_params}")
    print(f"Validation F1 Score (weighted): {best_score*100:.2f}%")
    print(f"{'─'*70}")
    return best_model, best_params, best_score, results_df


def tune_random_forest(X_train, y_train, X_val, y_val, param_grid, ssl_method_name, n_iter=30):
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV with F1 score (weighted).
    
    Uses 3-fold cross-validation and tests n_iter random combinations.
    """
    print(f"\n{'='*70}")
    print(f"TUNING RF (trained on {ssl_method_name} labels)")
    print(f"{'='*70}")
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_grid, n_iter=n_iter, cv=3,
        scoring='f1_weighted', random_state=42, n_jobs=-1, verbose=0
    )
    
    print(f"Running RandomizedSearchCV ({n_iter} iterations, 3-fold CV)...")
    print(f"Scoring: F1 Score (weighted)")
    random_search.fit(X_train, y_train)
    
    # Evaluate on validation set
    best_rf = random_search.best_estimator_
    y_val_pred = best_rf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n===> Tuning completed !")
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV F1 score: {random_search.best_score_*100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Validation F1 Score: {val_f1*100:.2f}%")
    
    return best_rf, random_search.best_params_, val_f1, pd.DataFrame(random_search.cv_results_)