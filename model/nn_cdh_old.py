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
            nn.Tanh(),
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


class NNCDHAdapter(nn.Module):
    """
    NN-CDH adaptation module, designed to plug into NN-KNN.

    Usage for training (offline, from pair data):
        adapter = NNCDHAdapter(feature_dim, label_dim)
        adapter.fit_pairs(X_cdh, y_cdh)

    Usage inside NN-KNN (in regression mode):
        adapted_labels = adapter(
            query_features,  # [B, D]
            case_features,   # [N, D]
            case_labels      # [N, label_dim] or [N]
        )
        # adapted_labels -> [B, N, label_dim]

    This matches the interface we wired into NN_KNN_Model:
        self.nn_cdh(query_features, case_features, case_labels)
    """

    def __init__(
        self,
        feature_dim: int,
        label_dim: int = 1,
        hidden_dims: Tuple[int, int] = (16, 8),
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.label_dim = label_dim

        input_dim = 2 * feature_dim  # [context, problem_diff]
        self.adapt_net = AdaptationModel(
            input_dim=input_dim,
            output_dim=label_dim,
            hidden_dims=hidden_dims,
        )

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
        pair_inputs_flat = pair_inputs.reshape(B * N, -1)     # [B*N, 2D]

        Δy_flat = self.adapt_net(pair_inputs_flat)            # [B*N, label_dim]
        Δy = Δy_flat.view(B, N, self.label_dim)               # [B, N, label_dim]

        base_labels = case_labels.view(1, N, self.label_dim).expand(B, N, self.label_dim)
        adapted_labels = base_labels + Δy                     # [B, N, label_dim]

        return adapted_labels

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
        Train the adaptation network on a pre-built pair dataset (X_cdh, y_cdh).
        Note: this is only used to train the adapter offline, from pair data. 

        X_cdh: [num_pairs, 2 * feature_dim]  (context, problem_diff)
        y_cdh: [num_pairs, label_dim]        (solution_diff)

        Returns:
            val_loss_history: list of validation losses per epoch.
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

        optimizer = torch.optim.Adam(self.adapt_net.parameters(), lr=1e-4)
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
                preds = self.adapt_net(xb)
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
                    preds = self.adapt_net(xb)
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
