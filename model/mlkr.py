import torch
import numpy as np
from metric_learn import MLKR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import inspect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _patch_metric_learn_for_sklearn_18():
    """
    metric-learn 0.7.0 still calls sklearn validators with `force_all_finite`.
    sklearn 1.8 removed that keyword in favor of `ensure_all_finite`.
    Patch metric_learn._util wrappers at runtime for compatibility.
    """
    try:
        import metric_learn._util as ml_util
        from sklearn.utils.validation import check_X_y as sk_check_X_y
        from sklearn.utils.validation import check_array as sk_check_array
    except Exception:
        return

    if "force_all_finite" in inspect.signature(sk_check_X_y).parameters:
        return

    def _compat_check_X_y(*args, **kwargs):
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return sk_check_X_y(*args, **kwargs)

    def _compat_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return sk_check_array(*args, **kwargs)

    ml_util.check_X_y = _compat_check_X_y
    ml_util.check_array = _compat_check_array

def run_mlkr_sklearn(X_train, y_train, X_val, y_val, cfg, y_mean_raw, y_std_raw):
    _patch_metric_learn_for_sklearn_18()
    
    X_train_np = X_train.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy().ravel() 
    X_val_np   = X_val.detach().cpu().numpy()
    
    y_val_z_np = y_val.detach().cpu().numpy().ravel()

    # D = X_train_np.shape[1]
    # if D > 500: 
    #     print(f"Feature dimension is high ({D}), applying PCA to reduce to 128 dims...")
    #     pca = PCA(n_components=128, random_state=42)
    #     X_train_np = pca.fit_transform(X_train_np)
    #     X_val_np = pca.transform(X_val_np)
    #     print(f"New shape: {X_train_np.shape}")
    
    print("Training MLKR...")
    mlkr = MLKR(
        n_components=cfg.get("n_components", None),
        max_iter=cfg.get("max_iter", 1000),
        tol=1e-4,
        random_state=42
    )
    mlkr.fit(X_train_np, y_train_np)

    print("Training KNN (with learnt metric)...")
    knn = KNeighborsRegressor(
        n_neighbors=cfg.get("k", 25),
        weights='distance', 
        metric=mlkr.get_metric(),
        algorithm='brute'
    )
    knn.fit(X_train_np, y_train_np)

    y_pred_z = knn.predict(X_val_np)

    if isinstance(y_mean_raw, torch.Tensor):
        raw_mean = y_mean_raw.item()
    else:
        raw_mean = y_mean_raw
        
    if isinstance(y_std_raw, torch.Tensor):
        raw_std = y_std_raw.item()
    else:
        raw_std = y_std_raw

    y_pred_raw = y_pred_z * raw_std + raw_mean
    
    y_true_raw = y_val_z_np * raw_std + raw_mean
    
    mse = mean_squared_error(y_true_raw, y_pred_raw)
    rmse = np.sqrt(mse)

    print(f"Validation RMSE [RAW] (MLKR) for {cfg.get('dataset_name', None)}: {rmse:.4f}")
    return rmse
