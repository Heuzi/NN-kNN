import os

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from reg_data_yu import Reg_data
# from model.nnknn_model import NN_KNN_Model, default_args, GlocalFeatureWeight, reg_locality_reg_loss
from model.nnknn_model import train_model, default_args

from model.nn_cdh import NNCDHAdapter, add_to_pair_list
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_ROOT = Path(os.getcwd())          # notebook's current folder as project root
DATA_ROOT    = PROJECT_ROOT / "datasets"
CHECKPOINTS  = PROJECT_ROOT / "checkpoints"
print("Running locally.")

# Make project importable and set CWD to project root for consistent relative paths
# Add PROJECT_ROOT to sys.path *before* importing from within the project
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

print("PROJECT_ROOT =", PROJECT_ROOT)
print("DATA_ROOT     =", DATA_ROOT)
print("CHECKPOINTS  =", CHECKPOINTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def get_feature_dim_old(case, feature_extractor):
    if feature_extractor is None:
        # Assuming last dimension is feature dimension
        return case.shape[-1] 
        # Alternatively, if you want the total number of elements (if one case is a multi-dimensional array):
        # return torch.prod(torch.tensor(case.shape)).item()
    else:
        return feature_extractor.feature_dim
    
    
def train_model_old(X_train, y_train, X_val, y_val, feature_extractor, cfg): # , glocal_fw_set_num=glocal_fw_set_num
    """
    Train an NN-kNN model using the provided train/validation split.

    Args:
        X_train: Training feature tensor.
        y_train: Training labels.
        X_val: Validation feature tensor.
        y_val: Validation labels.
        cfg: Configuration object or dict with training hyperparameters.

    Returns:
        best_accuracy: The best accuracy achieved during training.
        glocal_weightor: The trained global feature weightor.
    """
    # Move data to the appropriate device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Compute a robust global label scale (MAD) on y_train
    #preferred over std for robustness
    with torch.no_grad():
        ytr = y_train.float().to(device).view(-1)
        med = ytr.median()
        mad = (ytr - med).abs().median()
        global_sigma_y = (1.4826 * mad).clamp_min(1e-6).view(1,1)  # shape [1,1]
    # Create DataLoader for validation data
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    # DataLoader for batching
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True
    )
    feature_dim = get_feature_dim(X_train, feature_extractor)
    # Initialize the global feature weightor
    glocal_weightor = GlocalFeatureWeight(feature_dim, cfg["glocal_fw_set_num"])
    glocal_weightor.to(device)


    # -------------------------------------------------
    # Optional: manually nudge glocal weights before training
    # -------------------------------------------------
    if cfg.get("glocal_init_from_true", False):
        with torch.no_grad():
            alpha = cfg.get("glocal_init_alpha", 0.7)  # 0=no effect, 1=exact true weights

            # Prefer regime-specific weights if provided
            if "true_feature_weights_glocal" in cfg:
                init = cfg["true_feature_weights_glocal"].to(device)  # [S_true, D]
                S_true, D_true = init.shape
                S_cfg = cfg["glocal_fw_set_num"]

                assert D_true == feature_dim, \
                    f"true_feature_weights_glocal has D={D_true}, expected {feature_dim}"

                # If the number of glocal sets equals the number of regimes, align 1:1
                if S_cfg == S_true:
                    target = init
                else:
                    # Tile regimes to fill all glocal sets
                    reps = (S_cfg + S_true - 1) // S_true
                    target = init.repeat(reps, 1)[:S_cfg]   # [S_cfg, D]

                glocal_weightor.feature_weights.data[:S_cfg] = (
                    (1 - alpha) * glocal_weightor.feature_weights.data[:S_cfg]
                    + alpha * target
                )

            # Fallback: single-regime true weights
            elif "true_feature_weights" in cfg:
                init = cfg["true_feature_weights"].to(device)  # [D]
                assert init.shape[0] == feature_dim, \
                    f"true_feature_weights has D={init.shape[0]}, expected {feature_dim}"
                # Nudge only the first glocal set
                glocal_weightor.feature_weights.data[0] = (
                    (1 - alpha) * glocal_weightor.feature_weights.data[0]
                    + alpha * init
                )

        # Optional debug print:
        print("Initial glocal feature weights after init-from-true:")
        print(glocal_weightor.feature_weights.data)

    task = cfg.get("task_type", default_args["task_type"])
    if task == "classification":
        criterion = nn.CrossEntropyLoss()
        num_classes = len(torch.unique(y_train))
        ys = torch.nn.functional.one_hot(y_train, num_classes=num_classes)
    else:
        criterion = nn.MSELoss()
        ys = y_train.float().unsqueeze(1)  # [N, 1] for scalar regression
    adapter = None
    if cfg.get("use_nn_cdh", False) and task == "regression":
        adapter = NNCDHAdapter(feature_dim, label_dim=1).to(device)
        #pre-train adapter here
        if cfg.get("nn_cdh_pretrain", False):
            # 1) Get latent features for all training cases
            if feature_extractor is None:
                with torch.no_grad():
                    Z_train = X_train  # [N, D] already in feature space
            else:
                feature_extractor.eval()
                with torch.no_grad():
                    Z_train = feature_extractor(X_train)  # [N, D]
                feature_extractor.train()

            Z_np = Z_train.detach().cpu().numpy().astype(np.float32)
            y_np = y_train.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
            N, D = Z_np.shape

            # 2) Build (X_cdh, y_cdh) pairs: [context, problem_diff] -> solution_diff
            X_cdh = np.empty((0, 2 * D), dtype=np.float32)
            Y_cdh = np.empty((0, 1), dtype=np.float32)

            rng = np.random.default_rng(cfg.get("nn_cdh_pair_seed", 0))
            pairs_per_target = cfg.get("nn_cdh_pairs_per_case", 5)

            for j in range(N):
                # choose some source indices for this target j
                src_indices = rng.choice(N, size=pairs_per_target, replace=False)
                x_target = Z_np[j:j+1, :]      # [1, D]
                y_target = y_np[j:j+1, :]      # [1, 1]
                for i in src_indices:
                    x_source = Z_np[i:i+1, :]  # [1, D]
                    y_source = y_np[i:i+1, :]  # [1, 1]
                    X_cdh, Y_cdh = add_to_pair_list(
                        X_cdh, Y_cdh,
                        x_target, x_source,
                        y_target, y_source,
                    )

            # 3) Fit the adapter on this pair dataset
            adapter.fit_pairs(
                X_cdh, Y_cdh,
                device=device,
            )
    model = NN_KNN_Model(X_train, ys, feature_extractor=feature_extractor, glocal_weightor=glocal_weightor, nn_cdh= adapter, **cfg)

    model.to(device)

    # Separate parameters for different learning rates
    feature_extractor_params = list()
    if(feature_extractor is not None):
        feature_extractor_params = list(feature_extractor.parameters())
    glocal_weightor_params = list()
    if(glocal_weightor is not None):
        glocal_weightor_params = list(glocal_weightor.parameters())
    adapter_params = list()
    if(adapter is not None):
        adapter_params = list(adapter.parameters())
    #print out number of parameters here

    shared_params_ids = {id(param) for param in feature_extractor_params + glocal_weightor_params + adapter_params}
    # case_net_params = [param for case_net in model.case_nets for param in case_net.parameters() if id(param) not in shared_params_ids]
    case_net_params = [param for param in model.parameters() if id(param) not in shared_params_ids]
    print("Number of feature extractor parameters:", len(feature_extractor_params))
    for param in feature_extractor_params:
        print(param.shape)
    print("Number of glocal weightor parameters:", len(glocal_weightor_params))
    for param in glocal_weightor_params:
        print(param.shape)
    print("Number of adapter parameters:", len(adapter_params))
    for param in adapter_params:
        print(param.shape)    
    print("Number of case_net_params:", len(case_net_params))
    for param in case_net_params:
        print(param.shape)
    print("*****************")
    
    fearture_extractor_lr = cfg.get("feature_extractor_lr", default_args["feature_extractor_lr"])
    glocal_weightor_lr = cfg.get("glocal_weightor_lr", default_args["glocal_weightor_lr"])
    adapter_lr = cfg.get("adapter_lr", default_args["adapter_lr"])
    case_net_lr = cfg.get("case_net_lr", default_args["case_net_lr"])   
    optimizer = torch.optim.Adam([
        {'params': feature_extractor_params, 'lr': fearture_extractor_lr },
        {'params': glocal_weightor_params, 'lr': glocal_weightor_lr },
        {'params': adapter_params, 'lr': adapter_lr },
        {'params': case_net_params, 'lr': case_net_lr }
    ], weight_decay=1e-5)

    patience_counter = 0
    metric_for_model_select = 0
    best_val_loss = float('inf')
    best_found = False
    best_epoch = 0
    training_epochs = cfg.get("training_epochs", default_args["training_epochs"])
    eps_sigma_multiplier = cfg.get("eps_sigma_multiplier", 0.2)
    print(f"Training started for training_epochs epochs with batch size {cfg.get('batch_size')}")

    # ===== TRAIN =====
    for epoch in range(training_epochs):
        if best_found: break
        model.train()
        comps = {}
        
        # Track training loss for the current epoch
        train_loss_total = 0.0
        train_batch_count = 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100):
            optimizer.zero_grad()
            final_predictions, predicted_solution, pre_adapted_solution, topk_cases, topk_labels, topk_acts = model(X_batch)

            if task == "classification":
                base = criterion(final_predictions, y_batch)
                loss, comps = base, {}
            else:
                base = criterion(final_predictions.squeeze(1), y_batch.float())
                if cfg.get("regression_locality", False):
                    loss, comps = reg_locality_reg_loss(
                        base, y_batch, topk_labels, topk_acts, global_sigma_y, cfg, eps_sigma_multiplier
                    )
                else:
                    loss, comps = base, {}
            
            # Accumulate loss
            train_loss_total += loss.item() * X_batch.size(0)
            train_batch_count += X_batch.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        # optional: print last-batch comps
        if task != "classification" and cfg.get("regression_locality", False):
            print("Training loss components:")
            print(" | ".join([f"{k}: {v.item():.4f}" for k, v in comps.items()]))
        
        # Calculate Average Training Loss
        avg_train_loss = train_loss_total / max(1, train_batch_count)

        # ===== VALIDATE (early stopping on regularized objective) =====
        model.eval()
        val_loss_total = 0.0
        predicted_solution_list = []

        # neighbor metrics containers
        if task != "classification":
            WLD_vals, NA_hits, nDCG_vals = [], [], []
            eps_eval = eps_sigma_multiplier * global_sigma_y  # aligned with train

        val_loss_comp = {}
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_final_predictions, batch_predicted_solution, batch_pre_adapted_solution, b_idx, b_labels, b_weights = model(batch_X)

                if task == "classification":
                    base_val = criterion(batch_final_predictions, batch_y)
                    val_loss_batch, _ = base_val, {}
                else:
                    base_val = criterion(batch_final_predictions.squeeze(1), batch_y.float())
                    if cfg.get("regression_locality", False) and model.top_k_mode:
                        val_loss_batch, val_loss_comp = reg_locality_reg_loss(
                            base_val, batch_y, b_labels, b_weights, global_sigma_y, cfg, eps_sigma_multiplier
                        )
                    else:
                        val_loss_batch, _ = base_val, {}

                val_loss_total += val_loss_batch.item() * batch_X.size(0)

                predicted_solution_list.append(batch_predicted_solution)

                # neighbor metrics (reporting only)
                if task != "classification" and model.top_k_mode:
                    y_true  = batch_y.float().unsqueeze(1)
                    abs_d   = torch.abs(b_labels - y_true)

                    # normalized weights
                    epsw = torch.finfo(b_weights.dtype).eps
                    w = b_weights.clamp_min(epsw)
                    w = w / w.sum(dim=1, keepdim=True).clamp_min(epsw)

                    WLD_vals.append((w * abs_d).sum(dim=1))
                    K_use = b_weights.size(1)
                    NA_hits.append((abs_d <= eps_eval).float().mean(dim=1))

                    gains = torch.exp(-abs_d / (eps_eval + 1e-8))
                    denom = torch.log2(torch.arange(2, K_use + 2, device=gains.device).float())
                    dcg   = (gains[:, :K_use] / denom[:K_use]).sum(dim=1)
                    ideal = torch.sort(gains, dim=1, descending=True).values
                    idcg  = (ideal[:, :K_use] / denom[:K_use]).sum(dim=1).clamp_min(1e-8)
                    nDCG_vals.append(dcg / idcg)
        # optional: print last-batch comps
        if task != "classification" and cfg.get("regression_locality", False):
            print("Validation loss components:")
            print(" | ".join([f"{k}: {v.item():.4f}" for k, v in val_loss_comp.items()]))
        # aggregate
        predicted_solution = torch.cat(predicted_solution_list, dim=0)
        num_val = len(val_loader.dataset) if hasattr(val_loader, "dataset") else len(y_val)
        val_loss = val_loss_total / max(1, num_val)  # <-- early-stopping scalar

        log_dict = {}
        if cfg.get("use_wandb", False):
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss
            }

        if task == "classification":
            acc = accuracy_score(y_val.cpu().numpy(), predicted_solution.cpu().numpy())
            print(f"Epoch {epoch+1} - Val Acc: {acc:.4f} | Val Loss: {val_loss:.4f}")
            model_select_metric = acc
            if cfg.get("use_wandb", False):
                log_dict["val_acc"] = acc
        else:
            ss_res = torch.sum((y_val.float() - predicted_solution.float()) ** 2).item()
            ss_tot = torch.sum((y_val.float() - torch.mean(y_val.float())) ** 2).item()
            r2 = 1 - (ss_res / ss_tot)
            print(f"Epoch {epoch+1} - Val R²: {r2:.4f} | Reg Val Loss: {val_loss:.4f}")
            model_select_metric = r2
            if cfg.get("use_wandb", False):
                log_dict["val_r2"] = r2

            if model.top_k_mode:
                WLD  = torch.cat(WLD_vals).mean().item()
                NA   = torch.cat(NA_hits).mean().item()
                nDCG = torch.cat(nDCG_vals).mean().item()
                print(f"Neighbor metrics | WLD: {WLD:.4f} | NA@{K_use}: {NA:.3f} | nDCG: {nDCG:.3f}")
                if cfg.get("use_wandb", False):
                    log_dict.update({
                        "val_WLD": WLD,
                        "val_NA": NA,
                        "val_nDCG": nDCG
                    })
        
        if cfg.get("use_wandb", False):
            wandb.log(log_dict)

        checkpoint_path = cfg.get("checkpoint_path", default_args["checkpoint_path"])
        patience = cfg.get("patience", default_args.get("patience", 20))

        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best (epoch {epoch+1}) Reg Val Loss: {val_loss:.4f} — model saved.")
            patience_counter = 0
            metric_for_model_select = model_select_metric
            
            # Update best metrics in summary
            if cfg.get("use_wandb", False):
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            patience_counter += 1
            print(f"No improv. Best Reg Val Loss so far: {best_val_loss:.4f} (epoch {best_epoch+1})")

        if patience_counter > patience:
            print("Patience exceeded. Restoring best model.")
            model.load_state_dict(torch.load(checkpoint_path))
            best_found = True
            break
        
    if not cfg.get("save_ckpt", False):
        os.remove(checkpoint_path)

    print("Training completed. Best Acc or R2: ", metric_for_model_select)

    if cfg.get("use_wandb", False):
        wandb.run.summary["final_best_metric"] = metric_for_model_select

    glocal_weightor.project_feature_weights_inplace()
    print("Final global feature weights:", glocal_weightor.feature_weights)
    return metric_for_model_select, glocal_weightor, model

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2 python nnknn_reg.py --nn_cdh_pretrain --use_nn_cdh --save_ckpt --use_wandb --dataset califonia_housing
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bike_sharing")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--nn_cdh_pretrain", action="store_true")
    parser.add_argument("--use_nn_cdh", action="store_true")
    parser.add_argument("--save_ckpt", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset

    Xs, ys = Reg_data(dataset_name)   # Xs, ys are already standardized

    print(Xs.shape, ys.shape)
    print("y mean:", ys.mean().item(), "y std:", ys.std().item())
    print("Raw y min/max:", ys.min().item(), ys.max().item())
    
    X = Xs
    y = ys
    # Simple manual split (80/20)
    n_train = int(0.8 * X.size(0))
    indices = torch.randperm(X.size(0))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]  # y shape [N]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    print('Train:', X_train.shape, y_train.shape, '| Val:', X_val.shape, y_val.shape)
    

    # cfg = {
    #     **default_args,
    #     "dataset_name": dataset_name,
    #     "use_wandb": args.use_wandb,
    #     "nn_cdh_pretrain": args.nn_cdh_pretrain,
    #     "use_nn_cdh": args.use_nn_cdh,
    #     "save_ckpt": args.save_ckpt,
    #     "task_type": "regression",
    #     "softmax_over_cases": True,      # model enforces True in regression; kept explicit for clarity
    #     "tau": 1.0,                      # 1.0, attention sharpness over cases, lower -> sharper
    #     "case_normalizer": "sparsemax",  # 'softmax' | 'sparsemax' | 'entmax15'

    #     "post_mlp_enabled": False,
    #     "post_mlp_dims": (10, 8),
    #     "post_mlp_dropout": 0.0,
    #     "post_mlp_activation": "relu",

    #     #for locality regularization in regression
    #     "regression_locality": True,  # Whether to use locality regularization in regression training
    #     "lambda_base": 1.0,
    #     "lambda_kl": 0.0,        # weight for KL(p||w) attention-to-label proximity (regression only)
    #     "locality_alpha" : 2.0,    # exponent for distance in locality regularization
    #     "lambda_expdist": 0.3,   #0.3, weight for expected distance loss (regression only)
    #     "lambda_balance": 0.3,    #0.3, weight for signed-bias regularizer (regression only)
    #     "lambda_cover": 0.0,     #0.5, weight for coverage loss (regression
    #     "lambda_pair": 0.0,      #0.5, weight for pos>neg pairwise hinge (regression only)
    #     "lambda_ent": 0.0,      #1e-3, tiny entropy penalty (encourages sparse attention tails)
    #     "eps_sigma_multiplier": 0.1, #0.1,0.4 multiplier for sigma_y to define "close" and "far" in pairwise loss
    #     "pairwise_margin": 0.05, # required gap: w_pos - w_neg >= margin

    #     "pre_topk_mask": False,   # enable the pre-normalization K-sparsification, if turned on, this influences nn-cdh to only adapt to top_k neighbors
    #     "top_k": 40,         # K for Neighbor Agreement / nDCG in validation and also for explanation mode

    #     "glocal_fw_set_num": 1, #1
    #     "neg_weight_flag": False,        # regression should NOT use negative class weights
    #     "sampling_cases_flag": False,    # use all cases for stable regression
    #     "case_activation_by_top_k_average": True,  # use percentile-based bias finder
    #     "top_k_for_default_case_activation": 40, #mark.
    #     "case_activation_default_percentage": 0.1,
    #     "training_epochs": 1400,
    #     "batch_size": 64,
    #     "feature_extractor_lr": 1e-3, #1e-4,1e-3,
    #     "glocal_weightor_lr": 1e-3, #1e-3,
    #     "case_net_lr": 3e-5, #1e-5
    #     "checkpoint_path": f"./checkpoints/{dataset_name}_nnknn_reg.pth",
    #     "patience": 80,

    #     "explanation_mode": True
    # }

    #body fat best setting
    cfg = {
        **default_args,
        "dataset_name": dataset_name,
        "use_wandb": args.use_wandb,
        "nn_cdh_pretrain": args.nn_cdh_pretrain,
        "use_nn_cdh": args.use_nn_cdh,

        "cdh_aggregate": True,

        "save_ckpt": args.save_ckpt,
        "task_type": "regression",
        "softmax_over_cases": True,      # model enforces True in regression; kept explicit for clarity
        "tau": 1.0,                      # 1.0, attention sharpness over cases, lower -> sharper
        "case_normalizer": "sparsemax",  # 'softmax' | 'sparsemax' | 'entmax15'

        "post_mlp_enabled": False,
        "post_mlp_dims": (32, 16),
        "post_mlp_dropout": 0.1,
        "post_mlp_activation": "relu",

        #for locality regularization in regression
        "regression_locality": True,  # Whether to use locality regularization in regression training
        "lambda_base": 1.0,
        "lambda_kl": 0.0,        # weight for KL(p||w) attention-to-label proximity (regression only)
        "locality_alpha" : 2.0,    # exponent for distance in locality regularization
        "lambda_expdist": 0.3,   #0.3, weight for expected distance loss (regression only)
        "lambda_balance": 0.0,    #0.3, weight for signed-bias regularizer (regression only)
        "lambda_cover": 0.0,     #0.5, weight for coverage loss (regression
        "lambda_pair": 0.0,      #0.5, weight for pos>neg pairwise hinge (regression only)
        "lambda_ent": 0.0,      #1e-3, tiny entropy penalty (encourages sparse attention tails)
        "eps_sigma_multiplier": 0.1, #0.2,0.4 multiplier for sigma_y to define "close" and "far" in pairwise loss
        "pairwise_margin": 0.05, # required gap: w_pos - w_neg >= margin

        "pre_topk_mask": False,   # enable the pre-normalization K-sparsification, if turned on, this influences nn-cdh to only adapt to top_k neighbors
        "top_k": 20,         # K for Neighbor Agreement / nDCG in validation and also for explanation mode

        "glocal_fw_set_num": 1, #1
        "neg_weight_flag": False,        # regression should NOT use negative class weights
        "sampling_cases_flag": False,    # use all cases for stable regression
        "case_activation_by_top_k_average": True,  # use percentile-based bias finder
        "top_k_for_default_case_activation": 20, #mark.
        "case_activation_default_percentage": 0.1,
        "training_epochs": 3000,
        "batch_size": 64,
        "feature_extractor_lr": 1e-4, #1e-4,1e-3,
        "glocal_weightor_lr": 1e-3, #1e-3,
        "case_net_lr": 1e-5, #1e-5
        "checkpoint_path": f"./checkpoints/{dataset_name}_nnknn_reg.pth",
        "patience": 80,

        "explanation_mode": True
    }

    #body fat 0.58, 0.64, 0.68, 0.60
    # cfg = {
    #     **default_args,
    #     "dataset_name": dataset_name,
    #     "use_wandb": args.use_wandb,
    #     "nn_cdh_pretrain": args.nn_cdh_pretrain,
    #     "use_nn_cdh": args.use_nn_cdh,
    #     "save_ckpt": args.save_ckpt,
    #     "task_type": "regression",
    #     "softmax_over_cases": True,      # model enforces True in regression; kept explicit for clarity
    #     "tau": 1.0,                      # 1.0, attention sharpness over cases, lower -> sharper
    #     "case_normalizer": "sparsemax",  # 'softmax' | 'sparsemax' | 'entmax15'

    #     #for locality regularization in regression
    #     "regression_locality": True,  # Whether to use locality regularization in regression training
    #     "lambda_base": 1.0,
    #     "lambda_kl": 0.0,        # weight for KL(p||w) attention-to-label proximity (regression only)
    #     "locality_alpha" : 2.0,    # exponent for distance in locality regularization
    #     "lambda_expdist": 0.3,   #0.3, weight for expected distance loss (regression only)
    #     "lambda_balance": 0.3,    #0.3, weight for signed-bias regularizer (regression only)
    #     "lambda_cover": 0.0,     #0.5, weight for coverage loss (regression
    #     "lambda_pair": 0.0,      #0.5, weight for pos>neg pairwise hinge (regression only)
    #     "lambda_ent": 0.0,      #1e-3, tiny entropy penalty (encourages sparse attention tails)
    #     "eps_sigma_multiplier": 0.1, #0.2,0.4 multiplier for sigma_y to define "close" and "far" in pairwise loss
    #     "pairwise_margin": 0.05, # required gap: w_pos - w_neg >= margin

    #     "pre_topk_mask": False,   # enable the pre-normalization K-sparsification, if turned on, this influences nn-cdh to only adapt to top_k neighbors
    #     "top_k": 40,         # K for Neighbor Agreement / nDCG in validation and also for explanation mode

    #     "glocal_fw_set_num": 1, #1
    #     "neg_weight_flag": False,        # regression should NOT use negative class weights
    #     "sampling_cases_flag": False,    # use all cases for stable regression
    #     "case_activation_by_top_k_average": True,  # use percentile-based bias finder
    #     "top_k_for_default_case_activation": 40, #mark.
    #     "case_activation_default_percentage": 0.1,
    #     "training_epochs": 1400,
    #     "batch_size": 64,
    #     "feature_extractor_lr": 1e-4, #1e-4,1e-3,
    #     "glocal_weightor_lr": 1e-3, #1e-3,
    #     "case_net_lr": 1e-4, #1e-5
    #     "checkpoint_path": f"./checkpoints/{dataset_name}_nnknn_reg.pth",
    #     "patience": 80,

    #     "explanation_mode": True
    # }


    # cfg = {
    #     **default_args,
    #     "dataset_name": dataset_name,
    #     "use_wandb": args.use_wandb,
    #     "nn_cdh_pretrain": args.nn_cdh_pretrain,
    #     "use_nn_cdh": args.use_nn_cdh,
    #     "save_ckpt": args.save_ckpt,
    #     "task_type": "regression",
    #     "softmax_over_cases": True,      # model enforces True in regression; kept explicit for clarity
    #     "tau": 1.0,                      # attention sharpness over cases, lower -> sharper
    #     "case_normalizer": "sparsemax",  # 'softmax' | 'sparsemax' | 'entmax15'

    #     #for locality regularization in regression
    #     "regression_locality": True,  # Whether to use locality regularization in regression training
    #     "lambda_base": 1.0,
    #     "lambda_kl": 0.0,        # weight for KL(p||w) attention-to-label proximity (regression only)
    #     "locality_alpha" : 2.0,    # exponent for distance in locality regularization
    #     "lambda_expdist": 0.3,   #1.0, weight for expected distance loss (regression only)
    #     "lambda_balance": 0.3,    # weight for signed-bias regularizer (regression only)
    #     "lambda_cover": 0.0,     #0.5, weight for coverage loss (regression
    #     "lambda_pair": 0.0,      #0.5, weight for pos>neg pairwise hinge (regression only)
    #     "lambda_ent": 1e-3,  # 0.0,      #1e-3, tiny entropy penalty (encourages sparse attention tails)
    #     "eps_sigma_multiplier": 0.1, #0.2,0.4 multiplier for sigma_y to define "close" and "far" in pairwise loss
    #     "pairwise_margin": 0.05, # required gap: w_pos - w_neg >= margin

    #     "pre_topk_mask": False,   # enable the pre-normalization K-sparsification, if turned on, this influences nn-cdh to only adapt to top_k neighbors
    #     "top_k": 20,         # K for Neighbor Agreement / nDCG in validation and also for explanation mode

    #     "glocal_fw_set_num": 1, #1
    #     "neg_weight_flag": False,        # regression should NOT use negative class weights
    #     "sampling_cases_flag": False,    # use all cases for stable regression
    #     "case_activation_by_top_k_average": True,  # use percentile-based bias finder
    #     "top_k_for_default_case_activation": 40, #mark.
    #     "case_activation_default_percentage": 0.1,
    #     "training_epochs": 800,
    #     "batch_size": 64,
    #     "feature_extractor_lr": 1e-4, #1e-4,1e-3,
    #     "glocal_weightor_lr": 1e-3, #1e-3, influence feature weights and glocal weights
    #     "case_net_lr": 1e-4, #1e-4,1e-3,
    #     "checkpoint_path": f"./checkpoints/{dataset_name}_nnknn_reg.pth",
    #     "patience": 40,

    #     "explanation_mode": True
    # }

    if cfg['use_wandb']:
        run = wandb.init(
            entity="yuwang1-indiana-university",
            project="knn-reg",
            # name=f"{dataset_name}_nnknn_{'cdh' if cfg['use_nn_cdh'] else 'nocdh'}_{'pretrain' if cfg['nn_cdh_pretrain'] else 'nopretrain'}",
            name=f"{dataset_name}_knn",
            config=cfg
        )
    
    # No feature extractor for tabular demo
    feature_extractor = None
    # from model.weightedkNN import run_weighted_knn
    # rmse_post, mse_post = run_weighted_knn(X_train, y_train, X_val, y_val, feature_extractor, cfg)
    
    # from model.mlp import run_mlp
    # rmse_mlp = run_mlp(X_train, y_train, X_val, y_val, feature_extractor, cfg)
    # from model.mlkr import run_mlkr_sklearn
    # rmse_mlkr, mse_mlkr = run_mlkr_sklearn(X_train, y_train, X_val, y_val, feature_extractor, cfg)

    # from gbrt import run_gbrt_sklearn
    # rmse_gbrt, mse_gbrt = run_gbrt_sklearn(X_train, y_train, X_val, y_val, feature_extractor, cfg)
    
    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-6)
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm   = (y_val   - y_mean) / y_std

    # 2. Train
    best_acc, glocal_weightor, model = train_model(
        X_train, y_train_norm, X_val, y_val_norm,
        feature_extractor=feature_extractor,
        cfg=cfg
    )

    # 3. Single Evaluation Pass
    print('Best (R^2 proxy in logs).')

    model.eval()
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32)
    y_std_t  = torch.as_tensor(y_std,  dtype=torch.float32).clamp_min(1e-6)

    # Config: Is y_val in the loop already Z-scored?
    # Based on your snippet, y_val is likely the original raw tensor, so False.
    y_val_is_z = False 

    with torch.no_grad():
        pre_preds, post_preds = [], []
        B = 512
        for i in range(0, X_val.size(0), B):
            # forward returns: out, yhat (post), pre_adapt_yhat (pre), ...
            out, yhat, pre_adapt_yhat, *_ = model(X_val[i:i+B].to(device))

            # Sum over cases to get pre-adaptation prediction if needed
            # Assuming pre_adapt_yhat is [B, N_cases, 1]
            if pre_adapt_yhat.dim() == 3:
                pre_yhat = pre_adapt_yhat.sum(dim=1).squeeze(1)   # [B]
            elif pre_adapt_yhat.dim() == 2:
                pre_yhat = pre_adapt_yhat.squeeze(1)              # [B]
            else:
                pre_yhat = pre_adapt_yhat                          # already [B]

            post_preds.append(yhat.cpu())
            pre_preds.append(pre_yhat.cpu())

        # Concatenate results
        y_pred_post = torch.cat(post_preds, dim=0).view(-1)
        y_pred_pre  = torch.cat(pre_preds,  dim=0).view(-1)
        y_true      = y_val.view(-1).to(torch.float32)

    # 4. Un-normalize
    y_pred_post = y_pred_post * y_std_t + y_mean_t
    y_pred_pre  = y_pred_pre  * y_std_t + y_mean_t

    if y_val_is_z:
        y_true = y_true * y_std_t + y_mean_t

    # 5. Compute Metrics
    mse_pre  = F.mse_loss(y_pred_pre,  y_true).item()
    mse_post = F.mse_loss(y_pred_post, y_true).item()

    rmse_pre  = mse_pre ** 0.5
    rmse_post = mse_post ** 0.5
    delta_rmse = rmse_pre - rmse_post

    # 6. Print Everything
    print(f"Validation RMSE (Post): {rmse_post:.4f}") 
    print("-" * 30)
    print(f"Pre-adaptation  MSE:  {mse_pre:.6f}, RMSE: {rmse_pre:.6f}")
    print(f"Post-adaptation MSE:  {mse_post:.6f}, RMSE: {rmse_post:.6f}")
    print(f"ΔRMSE (pre - post):   {delta_rmse:.6f}")

    if cfg.get("use_wandb", False):
        wandb.log({
            "final_val_rmse_post": rmse_post,
            "final_val_rmse_pre": rmse_pre,
            "final_val_mse_post": mse_post,
            "final_val_mse_pre": mse_pre,
            "final_val_delta_rmse": delta_rmse
        })
        
        wandb.run.summary["final_val_rmse"] = rmse_post
        wandb.run.summary["final_delta_rmse"] = delta_rmse