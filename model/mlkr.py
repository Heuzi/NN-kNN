import torch
import numpy as np
from metric_learn import MLKR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_mlkr_sklearn(X_train, y_train, X_val, y_val, cfg, y_mean_raw, y_std_raw):
    
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