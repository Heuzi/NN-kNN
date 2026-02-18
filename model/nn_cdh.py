# nn_cdh.py
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class AdaptationModel(nn.Module):
    """
    Simple MLP for NN-CDH-style adaptation.
    Input: concatenated [context, problem_difference]
           where context = source/case features,
                 problem_difference = target/query - source.
    Output: predicted solution difference (Δy).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, int] = (16, 8),
    ):
        super().__init__()
        h1, h2 = hidden_dims
        self.model = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim),
            # As in your notebook: Tanh for regression/ classification-friendly bounded Δy
            # nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def add_to_pair_list(
    X_cdh: np.ndarray,
    Y_cdh: np.ndarray,
    x_target: np.ndarray,
    x_source: np.ndarray,
    y_target: np.ndarray,
    y_source: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy helper mirroring your notebook's addToPairList:

      X_cdh_pair = [x_source, x_target - x_source]
      y_cdh_pair = y_target - y_source

    Used for building the (context, problem_diff) -> solution_diff dataset.
    """
    X_cdh_pair = np.concatenate([x_source, x_target - x_source], axis=1)
    y_cdh_pair = y_target - y_source

    if X_cdh.size == 0:
        X_cdh = X_cdh_pair
    else:
        X_cdh = np.concatenate([X_cdh, X_cdh_pair], axis=0)

    if Y_cdh.size == 0:
        Y_cdh = y_cdh_pair
    else:
        Y_cdh = np.concatenate([Y_cdh, y_cdh_pair], axis=0)

    return X_cdh, Y_cdh

def add_to_aggregate_list(
    X_agg: np.ndarray,
    Y_agg: np.ndarray,
    x_target: np.ndarray,     # [1, D]
    y_target: np.ndarray,     # [1, label_dim]
    x_sources: np.ndarray,    # [K, D]
    y_sources: np.ndarray,    # [K, label_dim]
    weights: np.ndarray,      # [K] sums to 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build ONE aggregate-mode training example:

        x_bar = sum_i w_i * x_i
        y0    = sum_i w_i * y_i
        X     = [x_target - x_bar, y0]
        Y     = y_target - y0

    This matches forward_aggregate's feature design.
    """
    x_bar = (weights[:, None] * x_sources).sum(axis=0, keepdims=True)  # [1, D]
    y0    = (weights[:, None] * y_sources).sum(axis=0, keepdims=True)  # [1, label_dim]

    X_one = np.concatenate([x_target - x_bar, y0], axis=1)             # [1, D+label_dim]
    Y_one = y_target - y0                                              # [1, label_dim]

    if X_agg.size == 0:
        X_agg = X_one
    else:
        X_agg = np.concatenate([X_agg, X_one], axis=0)

    if Y_agg.size == 0:
        Y_agg = Y_one
    else:
        Y_agg = np.concatenate([Y_agg, Y_one], axis=0)

    return X_agg, Y_agg


class NNCDHAdapter(nn.Module):
    """
    NN-CDH adaptation module, designed to plug into NN-KNN.

    Backward-compatible behavior:
      - Calling adapter(query_features, case_features, case_labels) returns per-(query,case)
        adapted labels of shape [B, N, label_dim] (the original NN-CDH behavior).

    New (recommended) behavior for "single adaptation on the retrieval aggregate":
      - Call adapter.forward_aggregate(query_features, case_features, case_labels, case_weights)
        to obtain a single adapted prediction per query of shape [B, label_dim]:

            y0      = sum_i alpha_i * y_i
            x_bar   = sum_i alpha_i * x_i
            delta   = g([x_q - x_bar, y0])
            y_final = y0 + delta

        This keeps the adapter retrieval-conditioned while limiting capacity (one Δy per query).
    """

    def __init__(
        self,
        feature_dim: int,
        label_dim: int = 1,
        hidden_dims: Tuple[int, int] = (16, 8),
        hidden_dims_final: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.label_dim = label_dim

        # --- Per-pair adapter (original NN-CDH): Δy = f([x_source, x_target - x_source]) ---
        pair_input_dim = 2 * feature_dim
        self.adapt_net_pair = AdaptationModel(
            input_dim=pair_input_dim,
            output_dim=label_dim,
            hidden_dims=hidden_dims,
        )

        # --- Aggregate adapter (new): Δy = g([x_query - x_bar, y0]) ---
        if hidden_dims_final is None:
            hidden_dims_final = hidden_dims
        final_input_dim = feature_dim + label_dim
        self.adapt_net_final = AdaptationModel(
            input_dim=final_input_dim,
            output_dim=label_dim,
            hidden_dims=hidden_dims_final,
        )

    # -----------------------------
    # Feature builders
    # -----------------------------
    @staticmethod
    def build_pair_features(
        source_features: torch.Tensor,  # [..., D]
        target_features: torch.Tensor,  # [..., D]
    ) -> torch.Tensor:
        """
        Build [context, problem_diff] = [x_source, x_target - x_source].
        Broadcasts over leading dimensions as needed.
        """
        problem_diff = target_features - source_features
        return torch.cat([source_features, problem_diff], dim=-1)

    @staticmethod
    def build_final_features(
        query_features: torch.Tensor,   # [B, D]
        x_bar: torch.Tensor,            # [B, D]
        y0: torch.Tensor,               # [B, label_dim]
    ) -> torch.Tensor:
        """
        Build features for single-step, retrieval-conditioned adaptation:
            [x_query - x_bar, y0]
        """
        diff = query_features - x_bar
        return torch.cat([diff, y0], dim=-1)

    # -----------------------------
    # Original forward: per-(query,case) adaptation
    # -----------------------------
    def forward(
        self,
        query_features: torch.Tensor,   # [B, D]
        case_features: torch.Tensor,    # [N, D]
        case_labels: torch.Tensor,      # [N] or [N, label_dim]
    ) -> torch.Tensor:
        """
        Compute adapted labels for each (query, case) pair.

        Returns:
            adapted_labels: [B, N, label_dim]
        """
        device = next(self.parameters()).device
        query_features = query_features.to(device)
        case_features = case_features.to(device)
        case_labels = case_labels.to(device)

        if case_labels.dim() == 1:
            case_labels = case_labels.unsqueeze(-1)  # [N, 1]

        B, D = query_features.shape
        N = case_features.shape[0]
        assert D == self.feature_dim, (
            f"feature_dim mismatch: query_features has D={D}, "
            f"but adapter.feature_dim={self.feature_dim}"
        )

        # Expand to [B, N, D]
        target_exp = query_features.unsqueeze(1).expand(B, N, D)
        source_exp = case_features.unsqueeze(0).expand(B, N, D)

        pair_inputs = self.build_pair_features(source_exp, target_exp)  # [B, N, 2D]
        pair_inputs_flat = pair_inputs.reshape(B * N, -1)               # [B*N, 2D]

        dy_flat = self.adapt_net_pair(pair_inputs_flat)                  # [B*N, label_dim]
        dy = dy_flat.view(B, N, self.label_dim)                          # [B, N, label_dim]

        base_labels = case_labels.view(1, N, self.label_dim).expand(B, N, self.label_dim)
        adapted_labels = base_labels + dy                                # [B, N, label_dim]

        return adapted_labels

    # -----------------------------
    # New forward: single adaptation on retrieval aggregate
    # -----------------------------
    def forward_aggregate(
        self,
        query_features: torch.Tensor,   # [B, D]
        case_features: torch.Tensor,    # [N, D]
        case_labels: torch.Tensor,      # [N] or [N, label_dim]
        case_weights: torch.Tensor,     # [B, N]  (e.g., attention/normalized weights)
    ) -> torch.Tensor:
        """
        Perform a single, retrieval-conditioned adaptation per query.

        Args:
            query_features: [B, D]
            case_features:  [N, D]
            case_labels:    [N] or [N, label_dim]
            case_weights:   [B, N] (should sum to 1 over N; we do not enforce normalization here)

        Returns:
            y_final: [B, label_dim]
        """
        device = next(self.parameters()).device
        query_features = query_features.to(device)
        case_features = case_features.to(device)
        case_labels = case_labels.to(device)
        case_weights = case_weights.to(device)

        if case_labels.dim() == 1:
            case_labels = case_labels.unsqueeze(-1)  # [N, 1]

        B, D = query_features.shape
        N = case_features.shape[0]
        assert D == self.feature_dim, (
            f"feature_dim mismatch: query_features has D={D}, "
            f"but adapter.feature_dim={self.feature_dim}"
        )
        assert case_weights.shape == (B, N), (
            f"case_weights must be [B, N] = [{B}, {N}], got {tuple(case_weights.shape)}"
        )

        # Weighted prototype in feature space: x_bar = sum_i alpha_i x_i   -> [B, D]
        x_bar = case_weights @ case_features  # [B, D]

        # Base retrieval prediction: y0 = sum_i alpha_i y_i                -> [B, label_dim]
        y0 = case_weights @ case_labels       # [B, label_dim]

        # Single-step correction
        final_inputs = self.build_final_features(query_features, x_bar, y0)  # [B, D+label_dim]
        dy = self.adapt_net_final(final_inputs)                               # [B, label_dim]

        y_final = y0 + dy
        return y_final

    # -----------------------------
    # Offline pretraining (pair net only, unchanged)
    # -----------------------------
    def fit_pairs(
        self,
        X_cdh: np.ndarray,
        y_cdh: np.ndarray,
        epochs: int = 400,
        patience: int = 10,
        batch_size: int = 32,
        val_ratio: float = 0.1,
        verbose: bool = True,
        device: Optional[torch.device] = None,
    ) -> list:
        """
        Train the per-pair adaptation network on a pre-built pair dataset (X_cdh, y_cdh).

        NOTE: This trains `adapt_net_pair` (the original NN-CDH pair model):
            X_cdh: [num_pairs, 2 * feature_dim]  (context, problem_diff)
            y_cdh: [num_pairs, label_dim]        (solution_diff)

        The new aggregate adapter (`adapt_net_final`) is intended to be trained end-to-end
        inside NN-KNN (since it depends on retrieval weights/prototypes).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        X_cdh_tensor = torch.tensor(X_cdh, dtype=torch.float32)
        y_cdh_tensor = torch.tensor(y_cdh, dtype=torch.float32)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_cdh_tensor, y_cdh_tensor, test_size=val_ratio, shuffle=True
        )

        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_valid, y_valid)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size)

        optimizer = torch.optim.Adam(self.adapt_net_pair.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state_dict = None
        patience_counter = 0
        val_loss_history = []

        for epoch in range(epochs):
            # ---- Train ----
            self.train()
            train_loss_epoch = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                preds = self.adapt_net_pair(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item() * xb.size(0)

            train_loss_epoch /= len(train_loader.dataset)

            # ---- Validate ----
            self.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = self.adapt_net_pair(xb)
                    loss = criterion(preds, yb)
                    val_loss_epoch += loss.item() * xb.size(0)

            val_loss_epoch /= len(valid_loader.dataset)
            val_loss_history.append(val_loss_epoch)

            if verbose:
                print(
                    f"[NN-CDH] Epoch {epoch+1:4d} "
                    f"train_loss={train_loss_epoch:.6f} "
                    f"val_loss={val_loss_epoch:.6f}"
                )

            # ---- Early stopping ----
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print("[NN-CDH] Early stopping triggered.")
                    break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        if verbose and len(val_loss_history) > 0:
            print("[NN-CDH] Final val_loss:", val_loss_history[-1])

        return val_loss_history
    def fit_aggregate(
        self,
        X_agg: np.ndarray,
        y_agg: np.ndarray,
        epochs: int = 400,
        patience: int = 10,
        batch_size: int = 32,
        val_ratio: float = 0.1,
        verbose: bool = True,
        device: Optional[torch.device] = None,
    ) -> list:
        """
        Offline pretraining for aggregate mode.

        X_agg: [num_examples, feature_dim + label_dim]   where X = [x_target - x_bar, y0]
        y_agg: [num_examples, label_dim]                 where y = (y_target - y0) = Δy
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        X_t = torch.tensor(X_agg, dtype=torch.float32)
        y_t = torch.tensor(y_agg, dtype=torch.float32)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_t, y_t, test_size=val_ratio, shuffle=True
        )

        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_valid, y_valid)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size)

        optimizer = torch.optim.Adam(self.adapt_net_final.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience_ctr = 0
        val_hist = []

        for epoch in range(epochs):
            self.train()
            tr_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = self.adapt_net_final(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * xb.size(0)
            tr_loss /= len(train_loader.dataset)

            self.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = self.adapt_net_final(xb)
                    loss = criterion(preds, yb)
                    va_loss += loss.item() * xb.size(0)
            va_loss /= len(valid_loader.dataset)

            val_hist.append(va_loss)
            if verbose:
                print(f"[NN-CDH-AGG] Epoch {epoch+1:4d} train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    if verbose:
                        print("[NN-CDH-AGG] Early stopping triggered.")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        if verbose and len(val_hist) > 0:
            print("[NN-CDH-AGG] Final val_loss:", val_hist[-1])

        return val_hist

