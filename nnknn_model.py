import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

import os
import numpy as np

from model.nn_cdh import NNCDHAdapter, add_to_pair_list

debug_print_flag = False
def debug_print(*args, **kwargs):
    if debug_print_flag:
        print(*args, **kwargs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_args = {
    # Replace the following with actual values or pass them in at runtime

    "post_mlp_enabled": False,
    "post_mlp_dims": (32, 16),
    "post_mlp_dropout": 0.1,
    "post_mlp_activation": "relu",
    "post_mlp_flatten_after_base": True,

    "task_type": "classification",  # or "regression
    "softmax_over_cases": False,  # Whether to apply softmax over case activations
    "tau": 1.0,  # Temperature parameter for softmax over case activations, higher = softer
    "case_normalizer": "softmax",   # 'softmax' | 'sparsemax' | 'entmax15'

    #for locality regularization in regression
    "regression_locality": False,  # Whether to use locality regularization in regression training
    "lambda_kl": 0.0,        # weight for KL(p||w) attention-to-label proximity (regression only)
    "lambda_pair": 0.0,      # weight for pos>neg pairwise hinge (regression only)
    "lambda_ent": 0.0,      # tiny entropy penalty (encourages sparse attention tails)
    "pairwise_margin": 0.05, # required gap: w_pos - w_neg >= margin
    "pre_topk_mask": False,  # enable the pre-normalization K-sparsification, if turned on, this influences nn-cdh to only adapt to top_k neighbors
    "top_k": 40,         # K for Neighbor Agreement / nDCG in validation and also for explanation mode

    "explanation_mode": False,  # Whether to output top-k activated cases for explanations

    # "feature_extractor": None,  # e.g., a CNN for images or embedding for text
    "feature_dim": 128,  # Example placeholder, replace with actual value as needed
    "glocal_fw_set_num": 1,
    'training_epochs': 1000,
    "neg_weight_flag": False,

    "sampling_cases_flag": False,
    "use_sampling_cases_divisor": False,
    "sampling_cases_divisor": 100,
    "num_samples": 5000,

    # DESIGN DECISION
    "case_activation_by_top_k_average": False,
    "top_k_for_default_case_activation": 20,

    # if case_activation_by_top_k_average = False, following will be used
    "case_activation_default_percentage": 0.1,  

    # If true, will overwrite found default bias
    "bias_manual_set": True,
    "bias_manual_value": 0.0,
    "lambda_case_bias": 1e-3,

    "model_path": 'best_model.pth',
    # "feature_weightor_path": 'best_fw.pth',

    "ignore_identical_in_training": True, #effectively leave one out when retrieving cases in training
    "feature_extractor_lr": 5e-4,
    "glocal_weightor_lr": 5e-4,
    "adapter_lr": 1e-4,
    "case_net_lr": 1e-4,

    "nn_cdh_pretrain": False,  # Whether to pretrain nn_cdh adapter
    "use_nn_cdh": False,  # Whether to use nn_cdh for regression label adaptation
    "cdh_aggregate": True,

    "checkpoint_path": None,  # Path to save/load model checkpoints
}


def get_feature_dim(case, feature_extractor):
    if feature_extractor is None:
        # Assuming last dimension is feature dimension
        return case.shape[-1] 
        # Alternatively, if you want the total number of elements (if one case is a multi-dimensional array):
        # return torch.prod(torch.tensor(case.shape)).item()
    else:
        return feature_extractor.feature_dim
    
class GlocalFeatureWeight(nn.Module):
    def __init__(self, feature_dim, set_num):
        """
        Glocal feature weighting module for batched operations.

        Args:
            feature_dim: Dimensionality of the features in each GW set.
            set_num: Number of glocal weight sets.
        """
        super(GlocalFeatureWeight, self).__init__()
        self.feature_dim = feature_dim

        # Initialize feature weights
        self.feature_weights = nn.Parameter(torch.rand((set_num, feature_dim)), requires_grad=True)
        if set_num == 1:
            self.feature_weights = nn.Parameter(torch.ones((set_num, feature_dim)), requires_grad=True)  # Shape: (set_num, feature_dim)

    def forward(self, case_distance, glocal_weights):
        """
        Apply feature weighting to the case distance in a batched manner.

        Args:
            case_distance: Tensor of shape (batch_size, sample_num, feature_dim).
            glocal_weights: Tensor of shape (sample_num, set_num).
        """

        # Ensure positive feature weights using LeakyReLU
        pos_feature_weights = F.leaky_relu(self.feature_weights, negative_slope=0.001)  # (set_num, D)
        glocal_weights = F.leaky_relu(glocal_weights, negative_slope=0.001)             # (N_cases, set_num)

        # Compute weight factors for all cases
        weight_factors = torch.matmul(glocal_weights, pos_feature_weights)  # (N_cases, D)

        # Expand and multiply
        weighted_distance = case_distance * weight_factors.unsqueeze(0)     # (B, N_cases, D)
        return weighted_distance
    
    ## Alternative implementation with softplus and normalization
    # def forward(self, case_distance, glocal_weights):
    #     print("Inside GlocalFeatureWeight forward")
    #     eps = 1e-8

    #     # raw params
    #     raw_fw = self.feature_weights      # [S, D]
    #     raw_gw = glocal_weights           # [N_cases, S]

    #     # 1) Non-negative, normalized over D
    #     fw_pos  = F.softplus(raw_fw) + eps
    #     fw_norm = fw_pos / (fw_pos.sum(dim=-1, keepdim=True) + eps)   # [S, D]

    #     # 2) Non-negative, normalized over S
    #     gw_pos  = F.softplus(raw_gw) + eps
    #     gw_norm = gw_pos / (gw_pos.sum(dim=-1, keepdim=True) + eps)   # [N_cases, S]

    #     # 3) Per-case feature weights
    #     weight_factors = gw_norm @ fw_norm    # [N_cases, D]

    #     # 4) Apply to distances
    #     weighted_distance = case_distance * weight_factors.unsqueeze(0)
    #     return weighted_distance
    
    def get_feature_weights_display(self, detach: bool = True):
        """
        Display-only view of feature weights:
        - nonnegative
        - row-wise L1-normalized (each row sums to 1)
        Does NOT modify self.feature_weights.

        Args:
            detach: if True, returns a detached tensor for safe logging/printing.

        Returns:
            Tensor of shape [S, D] with nonnegative rows summing to 1.
        """
        eps = 1e-8
        fw = self.feature_weights

        # 1) Clamp to nonnegative (display only)
        fw_disp = fw.clamp_min(0.0)

        # 2) Normalize each row to sum to 1 (display only)
        row_sums = fw_disp.sum(dim=-1, keepdim=True).clamp_min(eps)
        fw_disp = fw_disp / row_sums

        return fw_disp.detach() if detach else fw_disp
# -----------------------------------------------------------------------------
# Optional post-feature MLP projector (useful for tabular / pooled embeddings)
# -----------------------------------------------------------------------------
class MLPFeatureProjector(nn.Module):
    """A small MLP feature projector (default: X -> 32 -> 16) with dropout.

    - Uses LazyLinear so input dim is inferred on first forward.
    - Intended for tabular data or already-pooled embeddings from an upstream extractor.
    - If upstream outputs non-2D tensors (e.g., images / sequences), this module flattens by default.
      For text models with sequence outputs, you should pool in the upstream extractor first.
    """

    def __init__(
        self,
        hidden_dims=(32, 16),
        dropout: float = 0.1,
        activation: str = "relu",
        flatten_input: bool = True,
    ):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        if len(self.hidden_dims) == 0:
            raise ValueError("MLPFeatureProjector requires at least one hidden dim (e.g., [32, 16]).")
        self.dropout = float(dropout)
        self.activation = activation.lower()
        self.flatten_input = bool(flatten_input)

        layers = []
        for hd in self.hidden_dims:
            layers.append(nn.LazyLinear(hd))
            if self.activation == "relu":
                layers.append(nn.ReLU())
            elif self.activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}. Use 'relu' or 'gelu'.")
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        self.net = nn.Sequential(*layers)

        # Expose an output dimension for downstream modules (e.g., GlocalFeatureWeight)
        self.feature_dim = int(self.hidden_dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten_input and x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class CompositeFeatureExtractor(nn.Module):
    """Compose an upstream feature extractor with an optional post-MLP projector."""

    def __init__(
        self,
        base_extractor: nn.Module | None,
        post_extractor: nn.Module | None,
        flatten_after_base: bool = True,
    ):
        super().__init__()
        self.base_extractor = base_extractor
        self.post_extractor = post_extractor
        self.flatten_after_base = bool(flatten_after_base)

        # best-effort feature_dim propagation
        self.feature_dim = None
        if self.post_extractor is not None and getattr(self.post_extractor, "feature_dim", None) is not None:
            self.feature_dim = int(getattr(self.post_extractor, "feature_dim"))
        elif self.base_extractor is not None and getattr(self.base_extractor, "feature_dim", None) is not None:
            self.feature_dim = int(getattr(self.base_extractor, "feature_dim"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_extractor is not None:
            x = self.base_extractor(x)
        if self.flatten_after_base and x.dim() > 2:
            x = x.view(x.size(0), -1)
        if self.post_extractor is not None:
            x = self.post_extractor(x)
        return x
    
def build_effective_feature_extractor(feature_extractor: nn.Module | None, cfg: dict) -> nn.Module | None:
    """Build the effective feature extractor pipeline.

    Supports:
      upstream extractor (e.g., CNN / text encoder) -> optional post-MLP projector -> NN-kNN core

    Controlled by cfg keys (defaults in default_args):
      - post_mlp_enabled: bool
      - post_mlp_dims: list/tuple, e.g. [32, 16]
      - post_mlp_dropout: float, e.g. 0.1
      - post_mlp_activation: 'relu' | 'gelu'
      - post_mlp_flatten_after_base: bool
    """
    enabled = cfg.get("post_mlp_enabled", default_args.get("post_mlp_enabled", False))
    if not enabled:
        return feature_extractor

    # Avoid double-wrapping if already composite with a post extractor
    if isinstance(feature_extractor, CompositeFeatureExtractor) and feature_extractor.post_extractor is not None:
        return feature_extractor

    post = MLPFeatureProjector(
        hidden_dims=cfg.get("post_mlp_dims", default_args.get("post_mlp_dims", (32, 16))),
        dropout=cfg.get("post_mlp_dropout", default_args.get("post_mlp_dropout", 0.1)),
        activation=cfg.get("post_mlp_activation", default_args.get("post_mlp_activation", "relu")),
        flatten_input=True,
    )
    return CompositeFeatureExtractor(
        base_extractor=feature_extractor,
        post_extractor=post,
        flatten_after_base=cfg.get("post_mlp_flatten_after_base", default_args.get("post_mlp_flatten_after_base", True)),
    )
# prompt: a class recording a case's How often it is activated, how often it is sampled, how often it correctly classifies a query.

class CaseRecord:
    def __init__(self, case_id = None):
        self.case_id = case_id
        self.activation_count = 0
        self.sample_count = 0
        self.correct_classification_count = 0

    def activate(self):
        self.activation_count += 1

    def sample(self):
        self.sample_count += 1

    def correct_classification(self):
        self.correct_classification_count += 1

    def get_activation_rate(self):
        return self.activation_count / (self.sample_count + 1e-10) # avoid division by zero

    def get_sampling_rate(self):
      return self.sample_count

    def get_accuracy(self):
        return self.correct_classification_count / (self.sample_count + 1e-10) # avoid division by zero


def find_default_bias_percentage(X, feature_extractor = None,num_samples=500, case_activation_default_percentage=0.1, distance_metric = F.pairwise_distance):
    """
    Estimates the default bias for CaseNets by randomly comparing pairwise distances.
    Handles both image (e.g., MNIST) and tabular data.

    Args:
        X: The feature tensor. Shape can be (num_cases, feature_dim) or (num_cases, 1, H, W).
        num_samples: The number of random case pairs to compare.
        case_activation_default_percentage: Percentage of sorted distances to select.

    Returns:
        The estimated default bias.
    """
    num_cases = X.shape[0]
    distances = []

    # Flatten if data is image-like (e.g., (num_cases, 1, 28, 28))
    if len(X.shape) > 2:
        X_flat = X.view(num_cases, -1)  # Flatten to (num_cases, feature_dim)
    else:
        X_flat = X  # Already in tabular form
    if feature_extractor is not None:
        # Extract features using the feature extractor
        with torch.no_grad():  # Disable gradient computation for efficiency
            X_flat = feature_extractor(X).view(X.shape[0], -1)  # Flatten to (num_cases, feature_dim)

    # Compute random pairwise distances
    for _ in range(num_samples):
        idx1, idx2 = torch.randint(0, num_cases, (2,))

        # Calculate the pairwise distance
        distance = distance_metric(X_flat[idx1].unsqueeze(0), X_flat[idx2].unsqueeze(0))
        distances.append(distance.item())

    # Sort distances and select top% based on case_activation_default_percentage
    distances.sort()
    percentile_index = int(len(distances) * case_activation_default_percentage)
    default_bias = distances[percentile_index]

    return default_bias


def find_default_bias_knn(X, feature_extractor=None, k=5, num_samples=500, batch_size=64, distance_metric = None):
    """
    Efficiently estimates the default bias for CaseNets by computing the average distance
    to the k-th nearest neighbor for each case, using a randomly sampled subset of cases.

    Args:
        X: The feature tensor, shape (num_cases, channels, height, width) for MNIST or CIFAR-10.
        feature_extractor: Optional feature extractor to transform the input tensor.
        k: The number of nearest neighbors to consider.
        num_samples: Number of random cases to compare against.
        batch_size: Batch size for processing cases.
        distance_metric: Optional custom distance metric function (has to be working for two collections of row vectors). If None, uses torch.cdist.

    Returns:
        The estimated default bias (average distance to the k-th nearest neighbor).
    """
    num_cases = X.shape[0]
    all_kth_distances = []

    # Extract features using the feature extractor, if provided
    if feature_extractor is not None:
        feature_list = []
        with torch.no_grad():
            for i in range(0, num_cases, batch_size):
                batch = X[i:i + batch_size]  # Shape: (batch_size, channels, height, width)
                batch_features = feature_extractor(batch)  # Shape: (batch_size, feature_dim)
                feature_list.append(batch_features)
        X = torch.cat(feature_list, dim=0)  # Concatenate all batches

    # Flatten the features for distance computation
    X_flat = X.view(X.shape[0], -1)  # Shape: (num_cases, feature_dim)

    # Randomly sample `num_samples` cases to form the comparison set
    sampled_indices = torch.randperm(num_cases)[:num_samples]
    sampled_cases = X_flat[sampled_indices]  # Shape: (num_samples, feature_dim)

    # Process cases in batches to reduce memory usage
    for i in range(0, num_cases, batch_size):
        # Get the batch of cases
        batch = X_flat[i:i + batch_size]  # Shape: (batch_size, feature_dim)

        # Compute pairwise distances between the batch and the sampled cases
        if distance_metric is None:
            distances = torch.cdist(batch, sampled_cases)  # Shape: (batch_size, num_samples)
        else:
            #NOT TESTED, problematic line
            distances = distance_metric(batch.unsqueeze(1), sampled_cases.unsqueeze(0))  # Shape: (batch_size, num_samples)

        # Set self-distances to infinity for sampled cases
        batch_indices = torch.arange(i, min(i + batch_size, num_cases))
        mask = (batch_indices.unsqueeze(1) == sampled_indices.unsqueeze(0))  # Match indices in the batch
        distances[mask] = float('inf')

        # Get the k-th smallest distance for each case in the batch
        k_actual = min(k, num_samples)  # Ensure k does not exceed the number of samples
        kth_distances = torch.topk(distances, k=k_actual, largest=False).values[:, -1]
        all_kth_distances.extend(kth_distances.tolist())

    # Return the average k-th neighbor distance as the default bias
    return sum(all_kth_distances) / len(all_kth_distances)


# --- Sparse/sparse-ish normalizers for case attention ---

import torch
import torch.nn.functional as F

def _view_along(z, dim, v):
    # reshape a 1D vector v so it can broadcast along axis=dim of z
    shape = [1] * z.dim()
    shape[dim] = -1
    return v.view(shape)

def sparsemax(z, dim=-1):
    """
    Sparsemax projection: Martins & Astudillo (2016).
    z: (..., n)  -> probabilities on simplex with exact zeros.
    """
    # sort scores descending
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)

    rhos = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    rhos = _view_along(z, dim, rhos)

    # Determine support: largest k s.t. z_sorted_k > (cumsum_k - 1) / k
    support = z_sorted > (z_cumsum - 1) / rhos
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)

    # Threshold
    tau = (z_cumsum.gather(dim, k - 1) - 1) / k

    # Projection
    p = (z - tau).clamp(min=0)

    # Numerically it already sums to ~1, but renorm for safety
    return p / (p.sum(dim=dim, keepdim=True) + 1e-12)

def entmax15(z, dim=-1, n_iters=50):
    """
    Entmax with alpha=1.5 (sparse, smoother than sparsemax).
    Uses bisection on tau so that sum(relu(z - tau)^2) == 1.
    """
    z_max = torch.amax(z, dim=dim, keepdim=True)
    z_min = torch.amin(z, dim=dim, keepdim=True)

    tau_lo = z_min - 1.0
    tau_hi = z_max

    def sqsum(tau):
        return torch.sum(torch.clamp(z - tau, min=0) ** 2, dim=dim, keepdim=True)

    target = torch.ones_like(z_max)

    # widen interval if needed
    for _ in range(10):
        too_small = (sqsum(tau_hi) > target)
        tau_hi = torch.where(too_small, tau_hi + (z_max - z_min + 1.0), tau_hi)
        too_large = (sqsum(tau_lo) < target)
        tau_lo = torch.where(too_large, tau_lo - (z_max - z_min + 1.0), tau_lo)
        if not (too_small.any() or too_large.any()):
            break

    # bisection
    for _ in range(n_iters):
        tau_mid = (tau_lo + tau_hi) / 2
        s_mid = sqsum(tau_mid)
        go_left = s_mid > target  # need larger tau to reduce S
        tau_lo = torch.where(go_left, tau_mid, tau_lo)
        tau_hi = torch.where(go_left, tau_hi, tau_mid)

    tau = (tau_lo + tau_hi) / 2
    p_unnorm = torch.clamp(z - tau, min=0) ** 2
    return p_unnorm / (p_unnorm.sum(dim=dim, keepdim=True) + 1e-12)

def normalize_cases(scores, normalizer="softmax", tau=1.0, dim=-1):
    """
    scores: pre-normalization case scores z (higher = better), shape [..., N]
    normalizer: 'softmax' | 'sparsemax' | 'entmax15'
    tau: temperature (pre-scales scores; lower => sharper/sparser)
    """
    s = scores / max(tau, 1e-8)
    if normalizer == "softmax":
        # stability shift only for softmax
        s = s - s.max(dim=dim, keepdim=True).values
        return torch.softmax(s, dim=dim)
    elif normalizer == "sparsemax":
        return sparsemax(s, dim=dim)
    elif normalizer == "entmax15":
        return entmax15(s, dim=dim)
    else:
        raise ValueError(f"Unknown normalizer: {normalizer}")


class NN_KNN_Model(nn.Module):
    def __init__(self, cases, labels, 
                 feature_extractor=None, 
                 feature_distance_metric = None, 
                 glocal_weightor=None, 
                 nn_cdh = None,
                  **kwargs):
        """
        Initializes the NN_KNN_Model.

        Args:
            cases (torch.Tensor): Tensor containing all cases (e.g., images or sequences).
            labels (torch.Tensor): Tensor of one-hot encoded labels for each case.
            feature_extractor: Optional feature extractor (e.g., CNN for images or embedding for text).
            feature_distance_metric: Optional custom distance metric function. If None, using euclidean distance.
            glocal_weightor: Optional glocal weightor for feature weighting.
            **kwargs: Additional configuration parameters that includes:
                - task_type: "classification" or "regression".
                - softmax_over_cases: Whether to apply softmax over case activations.
                - tau: Temperature parameter for softmax over case activations, higher = softer.
                - glocal_fw_set_num: Number of glocal weight sets.
                - neg_weight_flag: Whether to allow negative weights for negative classes
                - sampling_cases_flag: Whether to enable case sampling.
                - use_sampling_cases_divisor: Whether to use a divisor for sampling cases.
                - sampling_cases_divisor: (when use_sampling_cases_divisor true) Divisor for case sampling.
                - num_samples: (when use_sampling_cases_divisor false) Number of random cases, for retrieval and also for default bias calculation.
                - case_activation_by_top_k_average: Whether to set case activation based on top-k average distance.
                - top_k_for_default_case_activation: (when case_activation_by_top_k_average true) Number of top-k cases to consider for default bias calculation.
                - case_activation_default_percentage: (when case_activation_by_top_k_average false) Percentage of top cases to consider for default bias calculation.
                - bias_manual_set: Whether to manually set the case default bias.
                - bias_manual_value: (when bias_manual_set true) The manually set value for case default bias.
                - ignore_identical_in_training: Whether to leave one out in training when sampling cases.
                - top_k: (for explanation or for regression with locality regularization) K for Neighbor Agreement / nDCG in validation

        """

        super(NN_KNN_Model, self).__init__()
        self.cases = cases.to(device)  # Shape: [num_cases, *case_shape]
        self.labels = labels.to(device)  # Shape: [num_cases, num_classes]
        print("cases trainable:", self.cases.requires_grad)
        print("labels trainable:", self.labels.requires_grad)


        # self.labels = nn.Parameter(self.labels, requires_grad=False)  # Non-trainable
        self.feature_extractor = build_effective_feature_extractor(feature_extractor, kwargs)
        self.feature_distance_metric = feature_distance_metric
        self.glocal_weightor = glocal_weightor
        self.nn_cdh = nn_cdh

        self.adapt_enabled = True  # train_model will flip this during warm-up

        # load additional configuration parameters from kwargs
        self.config = kwargs
        self.task_type = kwargs.get('task_type', default_args['task_type'])
        self.softmax_over_cases = kwargs.get('softmax_over_cases', default_args['softmax_over_cases'])
        self.tau = kwargs.get('tau', default_args['tau'])

        if self.task_type == "regression": 
            self.softmax_over_cases = True  # enforce softmax over cases for regression
            print("Enforcing softmax_over_cases = True for regression task.")
        self.case_normalizer = kwargs.get('case_normalizer', default_args['case_normalizer'])

        self.glocal_weightor_set_num = kwargs.get('glocal_fw_set_num', default_args['glocal_fw_set_num'])
        self.neg_weight_flag = kwargs.get('neg_weight_flag', default_args['neg_weight_flag'])

        if self.neg_weight_flag and self.task_type != "classification":
            raise ValueError("neg_weight_flag can only be True for classification tasks. If you are doing regression, please set neg_weight_flag to False.")

        self.sampling_cases_flag = kwargs.get('sampling_cases_flag', default_args['sampling_cases_flag'])
        self.use_sampling_cases_divisor = kwargs.get('use_sampling_cases_divisor', default_args['use_sampling_cases_divisor'])
        self.sampling_cases_divisor = kwargs.get('sampling_cases_divisor', default_args['sampling_cases_divisor'])
        self.num_samples = kwargs.get('num_samples', default_args['num_samples'])   
        self.case_activation_by_top_k_average = kwargs.get('case_activation_by_top_k_average', default_args['case_activation_by_top_k_average'])
        self.top_k_for_default_case_activation = kwargs.get('top_k_for_default_case_activation', default_args['top_k_for_default_case_activation'])
        self.case_activation_default_percentage = kwargs.get('case_activation_default_percentage', default_args['case_activation_default_percentage'])
        self.bias_manual_set = kwargs.get('bias_manual_set', default_args['bias_manual_set'])
        self.bias_manual_value = kwargs.get('bias_manual_value', default_args['bias_manual_value'])
        self.model_path = kwargs.get('model_path', default_args['model_path'])  
        # self.feature_weightor_path = kwargs.get('feature_weightor_path', default_args['feature_weightor_path'])   
        self.ignore_identical_in_training = kwargs.get('ignore_identical_in_training', default_args['ignore_identical_in_training'])
        self.feature_extractor_lr = kwargs.get('feature_extractor_lr', default_args['feature_extractor_lr'])
        self.glocal_weightor_lr = kwargs.get('glocal_weightor_lr', default_args['glocal_weightor_lr'])
        self.case_net_lr = kwargs.get('case_net_lr', default_args['case_net_lr'])


        # Group cases by class, store their indices
        self.class_to_cases = {}
        if self.task_type == "classification":
            for i, label in enumerate(self.labels):
                class_label = torch.argmax(label).item()  # Extract class label
                if class_label not in self.class_to_cases:
                    self.class_to_cases[class_label] = []
                self.class_to_cases[class_label].append(i)
        else:
            # For regression, treat all cases as belonging  to the same "class". A single bucket.
            self.class_to_cases[0] = list(range(len(self.cases)))

        case_default_bias = 0
        if self.bias_manual_set:
            case_default_bias = self.bias_manual_value
        elif self.case_activation_by_top_k_average:
            case_default_bias = find_default_bias_knn(cases, feature_extractor, self.top_k_for_default_case_activation, self.num_samples)
        else:
            case_default_bias = find_default_bias_percentage(cases, feature_extractor, self.num_samples, self.case_activation_default_percentage)
        # Store the case_default_bias as a fixed (non-trainable) value
        self.case_default_bias = float(case_default_bias)

        # Parameters specific to each case
        # All cases have the same initial range
        self.biases = nn.Parameter(torch.full((len(cases),), case_default_bias))  # Shape: [num_cases]
        ### BIG decision, removing case weights
        # All cases have the same initial weight for their corresponding classes
        # self.weights = nn.Parameter(torch.ones(len(cases)))  # Shape: [num_cases]

        self.negative_weights = nn.Parameter(torch.ones(len(cases)))  # Shape: [num_cases]

        # Each case initially uses equal portion of all GW weights
        self.glocal_weights = nn.Parameter(
            torch.softmax(torch.ones(len(cases), self.glocal_weightor_set_num), dim=-1)
        )  # Shape: [num_cases, set_dim]

        # Precompute feature dimensions if feature extractor exists
        self.feature_dim = None
        if feature_extractor is not None:
            with torch.no_grad():
                dummy_input = cases[0].unsqueeze(0).to(device)
                self.feature_dim = feature_extractor(dummy_input).shape[-1]
        else:
            self.feature_dim = cases.shape[-1]
        self.cached_features = None  # To cache features during evaluation mode

        self.top_k_mode = False
        self.top_k = kwargs.get('top_k', default_args['top_k'])

        if (kwargs.get('regression_locality', default_args['regression_locality']) and self.task_type == "regression") or kwargs.get('explanation_mode', default_args['explanation_mode']):
            self.top_k_mode = True

    def mirrored_leaky_relu(self, x, negative_slope= 0.01 ):
        """
        Custom Leaky ReLU with mirrored behavior above a threshold.

        Args:
            x: Input tensor.
            negative_slope: Slope for x < 0.

        Returns:
            Transformed tensor.
        """
        threshold = self.case_default_bias
        # Leaky ReLU for x < 0
        leaky_part = torch.where(x < 0, negative_slope * x, x)

        # Mirroring effect for x > threshold
        mirror_part = torch.where(x > threshold,
                                threshold + negative_slope * (x - threshold),
                                leaky_part)

        return mirror_part

    def scaled_sigmoid(self, x):
        """
        Scaled and shifted sigmoid as a PyTorch operation.

        Args:
            x: The input tensor. the pre-activation value of a case
            A: The value at which the output should be close to 1.

        Returns:
            The scaled and shifted sigmoid output tensor.
        """
        A = self.case_default_bias
        s = 8 / A  # Scaling factor
        b = 0 - 4      # Shift value
        return torch.sigmoid(s * x + b)


    def _extract_features(self, case_indices):
        """
        Extract features for selected cases using the feature extractor.

        Args:
            case_indices (torch.Tensor): Indices of cases to process.

        Returns:
            extracted_features (torch.Tensor): Features for the selected cases.
        """
        selected_cases = self.cases[case_indices]  # Shape: [num_selected_cases, *case_shape]

        if self.feature_extractor is not None:
            if self.training:
                # Always compute features during training
                extracted_features = self.feature_extractor(selected_cases)  # Shape: [num_selected_cases, feature_dim]
                #wipe cache because feature extractor will be updated
                self.cached_features = None
            else:
                # During evaluation, update cache only for processed indices
                if self.cached_features is None:
                    # Initialize cache on the first evaluation pass
                    self.cached_features = torch.zeros(
                        (len(self.cases), self.feature_dim), dtype=torch.float32, device=selected_cases.device # type: ignore
                    ) # type: ignore

                # Check which indices need to be computed
                uncached_indices = [idx.item() for idx in case_indices if self.cached_features[idx].sum() == 0]
                if uncached_indices:
                    uncached_cases = self.cases[uncached_indices]
                    uncached_features = self.feature_extractor(uncached_cases)  # Extract features for uncached cases
                    self.cached_features[uncached_indices] = uncached_features

                # Retrieve features from the cache
                extracted_features = self.cached_features[case_indices]
        else:
            # No feature extraction; use raw cases as features
            extracted_features = selected_cases

        return extracted_features

    # def _adapt_labels_for_regression(
    #     self,
    #     query_features: torch.Tensor,   # [B, D]
    #     case_features: torch.Tensor,    # [N_sel, D]
    #     case_labels: torch.Tensor       # [N_sel, 1] or [N_sel]
    # ) -> torch.Tensor:
    #     """
    #     Use nn_cdh (if present) to adapt labels for each (query, case) pair.

    #     Expected nn_cdh interface:
    #         adapted = nn_cdh(
    #             query_features,  # [B, D]
    #             case_features,   # [N_sel, D]
    #             case_labels      # [N_sel, 1] or [N_sel]
    #         )
    #     and returns:
    #         adapted_labels: [B, N_sel, 1]

    #     If nn_cdh is None, this just tiles the raw case_labels to [B, N_sel, 1].
    #     """
    #     B = query_features.size(0)
    #     N_sel = case_features.size(0)

    #     # ensure shape [N_sel, 1]
    #     if case_labels.dim() == 1:
    #         case_labels = case_labels.unsqueeze(1)

    #     if self.nn_cdh is None:
    #         # No adaptation → broadcast raw labels over batch
    #         return case_labels.unsqueeze(0).expand(B, N_sel, 1)

    #     # With adaptation: delegate to nn_cdh
    #     adapted = self.nn_cdh(
    #         query_features,   # [B, D]
    #         case_features,    # [N_sel, D]
    #         case_labels       # [N_sel, 1]
    #     )

    #     # sanity check / reshape
    #     if adapted.dim() == 2:
    #         adapted = adapted.unsqueeze(-1)          # [B, N_sel, 1]
    #     return adapted



    def forward(self, query):
        """
        Perform forward pass and optionally provide explanations.

        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, *query_shape].

        Returns:
            final_predictions (torch.Tensor): Predicted probabilities/logits for each class.
            predicted_solution (torch.Tensor): Predicted class indices (classification) or values (regression).
            most_activated_cases (list, optional): List of top-k most activated cases (if explanation_mode=True).
            most_activated_case_labels (list, optional): Labels of the top-k most activated cases.
            most_activated_activations (torch.Tensor, optional): Activations of the top-k most activated cases.
        """
        batch_size = query.size(0)
        num_cases = len(self.cases)
        case_indices = torch.arange(num_cases).to(query.device)  # Default: use all case_nets
        # Sampling cases
        if self.sampling_cases_flag:
          sampled_indices = []
          each_class_sample_num = max(1, self.num_samples // len(self.class_to_cases))

          for class_label, class_case_indices in self.class_to_cases.items():
              if len(class_case_indices) >= each_class_sample_num:
                  # Sample directly from global indices
                  sampled_indices.extend(torch.tensor(class_case_indices)[torch.randperm(len(class_case_indices))[:each_class_sample_num]].tolist())
              else:
                  # If fewer cases, sample with replacement
                  sampled_indices.extend(
                      torch.tensor(class_case_indices)[torch.randint(0, len(class_case_indices), (each_class_sample_num,))].tolist()
                  )
          case_indices = torch.tensor(sampled_indices).to(query.device)

        # Extract features
        query_features = self.feature_extractor(query) if self.feature_extractor is not None else query
        case_features = self._extract_features(case_indices)

        # Compute distances
        query_expanded = query_features.unsqueeze(1).expand(-1, len(case_indices), -1)  # [batch_size, num_selected_cases, feature_dim]
        case_expanded = case_features.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_selected_cases, feature_dim]
        if self.feature_distance_metric is None:
            elementwise_distance = (query_expanded - case_expanded) ** 2  # Shape: [batch_size, num_selected_cases, feature_dim]
        else:
            elementwise_distance = self.feature_distance_metric(query_expanded, case_expanded)  # Shape: [batch_size, num_selected_cases, feature_dim]

        # Apply global-local weighting if applicable
        if self.glocal_weightor is not None:
            glocal_weights = self.glocal_weights[case_indices]  # [num_selected_cases, set_dim]
            elementwise_distance = self.glocal_weightor(elementwise_distance, glocal_weights)  # Weighted distances

        distances = torch.sqrt(torch.relu(torch.sum(elementwise_distance, dim=-1)))  # [batch_size, num_selected_cases]
        if self.ignore_identical_in_training and self.training:
            eps = 1e-8
            identical_mask = (distances < eps)
        else:
            identical_mask = None

        # Convert distances to activations
        # pre_activations = self.scaled_sigmoid(self.biases[case_indices] - distances)  # [batch_size, num_selected_cases]
        # weighted_activations = pre_activations * F.relu(self.weights[case_indices])  # Scale by case-specific weights
        # weighted_activations = self.biases[case_indices].unsqueeze(0) - distances
        # -----------------------------
        # Case scoring (case_score_mode)
        # -----------------------------
        mode = self.config.get("case_score_mode", "bias_minus_distance")
        if mode == "neg_distance":
            # KNN-like: closer => larger score
            z = -distances

        elif mode == "bias_minus_distance":
            # KNN-like with a per-case offset
            b = self.biases[case_indices].unsqueeze(0)          # [1, N_sel]
            z = b - distances                                   # [B, N_sel]
 
        elif mode == "neg_distance_logw":
            # KNN-like with a per-case "reliability" weight
            w = F.relu(self.weights[case_indices]).unsqueeze(0) # [1, N_sel]
            z = -distances + torch.log(w + 1e-8)                # [B, N_sel]

        elif mode == "sigmoid":
            # Your original style (but *as logits* for the normalizer)
            b = self.biases[case_indices].unsqueeze(0)          # [1, N_sel]
            # elementwise gate in (0,1); treat it as "score" for the normalizer
            z = self.scaled_sigmoid(b - distances)              # [B, N_sel]
            # If you want to include weights here, do it additively as logits:
            # w = F.relu(self.weights[case_indices]).unsqueeze(0)
            # z = z * w   # (not great for KNN-like behavior, but allowed)

        elif mode == "hard_knn":
            # Build weights directly: uniform over K nearest by distance
            K = int(self.config.get("top_k", 20))
            K = min(K, distances.size(1))

            dist2 = distances
            if identical_mask is not None:
                dist2 = dist2.masked_fill(identical_mask, float("inf"))

            # indices of K smallest distances (use -dist for topk)
            _, top_idx = torch.topk(-dist2, k=K, dim=1)

            weights = torch.zeros_like(distances)
            weights.scatter_(1, top_idx, 1.0 / K)

            # safety renorm if identicals existed
            if identical_mask is not None:
                weights = weights.masked_fill(identical_mask, 0.0)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

            weighted_activations = weights
            # z = None  # skip normalizer below

        else:
            raise ValueError(f"Unknown case_score_mode: {mode}")
        # -----------------------------
        # Normalize over cases if needed
        # -----------------------------
        if self.softmax_over_cases:
            if mode != "hard_knn":
                # z already computed above

                # Pre-mask for normalizer
                if identical_mask is not None:
                    if self.case_normalizer == "softmax":
                        z = z.masked_fill(identical_mask, float("-inf"))
                    else:
                        z = z.masked_fill(identical_mask, -1e9)

                # Optional top-k mask before normalization
                if self.config.get("pre_topk_mask", False):
                    K = int(self.config.get("top_k", 20))
                    K = min(K, z.size(1))
                    top_vals, top_idx = torch.topk(z, k=K, dim=1)
                    fill_val = float("-inf") if self.case_normalizer == "softmax" else -1e9
                    z_masked = torch.full_like(z, fill_val)
                    z = z_masked.scatter(1, top_idx, top_vals)

                weights = normalize_cases(z, normalizer=self.case_normalizer, tau=self.tau, dim=1)

                # Post-mask + renorm
                if identical_mask is not None:
                    weights = weights.masked_fill(identical_mask, 0.0)
                    denom = weights.sum(dim=1, keepdim=True)
                    all_zero = (denom <= 0)
                    if all_zero.any():
                        allowed = (~identical_mask).float()
                        allowed_sum = allowed.sum(dim=1, keepdim=True).clamp_min(1.0)
                        uniform = allowed / allowed_sum
                        weights = torch.where(all_zero, uniform, weights)
                        denom = weights.sum(dim=1, keepdim=True)
                    weights = weights / denom.clamp_min(1e-12)

                weighted_activations = weights
            # else: hard_knn already produced weighted_activations
        else:
            # no across-case normalizer; raw activations would be used (can be risky if negative!)
            # if you keep this path, ensure nonneg + renorm
            if mode != "hard_knn":
                weighted_activations = z
            if identical_mask is not None:
                weighted_activations = weighted_activations.masked_fill(identical_mask, 0.0)
        # Multiply activations by labels
        selected_labels = self.labels[case_indices]  # [num_selected_cases, num_classes]

        if self.task_type == "classification":
            labeled_activations = weighted_activations.unsqueeze(2) * selected_labels.unsqueeze(0)  # [batch_size, num_selected_cases, num_classes]

            if self.neg_weight_flag:
                print("ERROR: Negative weights for negative classes not yet implemented in this version.")
                # negative_labels =  (selected_labels - 1)
                # weighted_neg_activations =  (pre_activations * self.negative_weights[case_indices])
                # weighted_neg_activations = weighted_neg_activations.unsqueeze(-1) * negative_labels
                # labeled_activations = labeled_activations + weighted_neg_activations

            # Sum over cases to produce predictions
            # pre_adapted_solution = None  # Not used for classification
            final_predictions = labeled_activations.sum(dim=1)  # [batch_size, num_classes]
            predicted_solution = final_predictions.argmax(dim=1)  # [batch_size]
        else:  # regression
            # Ensure labels are [N_sel, 1]
            if selected_labels.dim() == 1:
                selected_labels_reg = selected_labels.unsqueeze(1)
            else:
                selected_labels_reg = selected_labels.view(-1, 1)   # [num_selected_cases, 1]
            # --- NN-CDH adaptation hook ---
            # IMPORTANT:
            # - "individual" mode: adapt each retrieved case (your old behavior)
            # - "aggregate" mode: adapt ONCE after aggregating retrieved cases (new behavior)
            #
            # Toggle with config:
            #   self.config.get("cdh_aggregate", True)
            #
            # Assumed tensors in scope:
            #   query_features         : [B, D]
            #   case_features          : [N_sel, D]
            #   selected_labels_reg    : [N_sel, 1]   (or [N_sel] but here it's treated as [N_sel,1])
            #   weighted_activations   : [B, N_sel]   (already softmax-normalized in regression)

            cdh_aggregate = bool(self.config.get("cdh_aggregate", True))

            # Base "no adaptation" labels: [B, N_sel, 1]
            base_labels = selected_labels_reg.unsqueeze(0).expand(batch_size, -1, 1)

            if (self.nn_cdh is not None) and getattr(self, "adapt_enabled", True):
                if cdh_aggregate:
                    # ===== NEW: single aggregate adaptation =====
                    # Use existing tensors (no undefined *_topk variables)
                    alpha = weighted_activations  # [B, N_sel]

                    # Final prediction directly from adapter: [B, 1]
                    final_predictions = self.nn_cdh.forward_aggregate(
                        query_features=query_features,     # [B, D]
                        case_features=case_features,       # [N_sel, D]
                        case_labels=selected_labels_reg,   # [N_sel, 1]
                        case_weights=alpha,                # [B, N_sel]
                    )  # -> [B, 1]

                    predicted_solution = final_predictions.squeeze(1)  # [B]

                    # For bookkeeping / analysis: "pre-adapted" prediction y0 = sum_i alpha_i * y_i
                    pre_adapted_solution = (alpha.unsqueeze(2) * base_labels).sum(dim=1)  # [B, 1]

                    # If downstream code expects adapted_labels, we can define a placeholder:
                    # It's not used for prediction in aggregate mode, but keeps debug prints safe.
                    adapted_labels = base_labels  # [B, N_sel, 1]

                else:
                    # ===== OLD: per-case adaptation =====
                    if self.config.get("pre_topk_mask", False):
                        # Only adapt the top-K neighbors (the ones with nonzero weights)
                        K = int(self.config.get("top_k", 20))
                        K = min(K, weighted_activations.size(1))  # safety

                        # top-K after normalization, aligned with gradient flow
                        top_w, top_idx = torch.topk(weighted_activations, k=K, dim=1)  # [B, K]

                        # Expand case features: [N_sel, D] -> [B, N_sel, D]
                        case_feat_exp = case_features.unsqueeze(0).expand(batch_size, -1, -1)
                        label_exp     = base_labels  # [B, N_sel, 1]

                        batch_idx = torch.arange(batch_size, device=query_features.device).unsqueeze(1)  # [B, 1]

                        top_case_features = case_feat_exp[batch_idx, top_idx]  # [B, K, D]
                        top_case_labels   = label_exp[batch_idx, top_idx]      # [B, K, 1]

                        # Build pair inputs: [x_source, x_target - x_source]
                        B, K_eff, D = top_case_features.shape
                        q_rep = query_features.unsqueeze(1).expand(B, K_eff, D)  # [B, K, D]

                        pair_inputs = self.nn_cdh.build_pair_features(
                            source_features=top_case_features,
                            target_features=q_rep
                        )  # [B, K, 2D]

                        pair_flat = pair_inputs.reshape(B * K_eff, -1)           # [B*K, 2D]
                        dy_flat   = self.nn_cdh.adapt_net(pair_flat)             # [B*K, 1]
                        dy        = dy_flat.view(B, K_eff, self.nn_cdh.label_dim)  # [B, K, 1]

                        adapted_topk = top_case_labels + dy  # [B, K, 1]

                        adapted_labels = base_labels.clone()                       # [B, N_sel, 1]
                        adapted_labels[batch_idx, top_idx, :] = adapted_topk

                    else:
                        # Full per-case adaptation over all selected cases
                        adapted_labels = self.nn_cdh(
                            query_features=query_features,       # [B, D]
                            case_features=case_features,         # [N_sel, D]
                            case_labels=selected_labels_reg      # [N_sel, 1]
                        )  # -> [B, N_sel, 1]

                    # Original aggregation logic (weighted sum over adapted labels)
                    pre_adapted_solution = weighted_activations.unsqueeze(2) * base_labels        # [B, N_sel, 1]
                    labeled_activations  = weighted_activations.unsqueeze(2) * adapted_labels     # [B, N_sel, 1]
                    final_predictions    = labeled_activations.sum(dim=1)                          # [B, 1]
                    predicted_solution   = final_predictions.squeeze(1)                            # [B]

            else:
                # ===== No adapter at all =====
                adapted_labels = base_labels

                pre_adapted_solution = weighted_activations.unsqueeze(2) * base_labels  # [B, N_sel, 1]
                labeled_activations  = weighted_activations.unsqueeze(2) * adapted_labels
                final_predictions    = labeled_activations.sum(dim=1)                   # [B, 1]
                predicted_solution   = final_predictions.squeeze(1)                     # [B]

    

        # ---------------- Top-K outputs (formerly "Explanation") ----------------
        # Backward-compat: if an older config set `explanation_mode`, honor it
    
        use_topk = getattr(self, "top_k_mode", False) or getattr(self, "explanation_mode", False)

        most_activated_cases = None
        most_activated_case_labels = None
        most_activated_activations = None

        if use_topk:
            # top-K by the normalized attention over selected cases
            top_k_activations, top_k_indices = torch.topk(
                weighted_activations, k=self.top_k, dim=1
            )  # [B, K]

            # Map selected-set indices -> global case indices
            # case_indices: [N_sel]; top_k_indices: [B, K] -> broadcasted gather to [B, K]
            topk_global_indices = case_indices[top_k_indices]  # [B, K] (long)

            # Labels as [B, K] tensor
            if isinstance(self.labels, torch.Tensor):
                # self.labels may be shape [N] or [N,1]; squeeze last dim if present
                topk_labels = self.labels[topk_global_indices]
                if topk_labels.dim() == 3 and topk_labels.size(-1) == 1:
                    topk_labels = topk_labels.squeeze(-1)
            else:
                # Fallback if labels are not a tensor (e.g., numpy); convert once
                topk_labels = torch.as_tensor(
                    self.labels[topk_global_indices.cpu().numpy()], device=weighted_activations.device
                )

            # (Optional) cases as [B, K, D] tensor when self.cases is a tensor
            if isinstance(self.cases, torch.Tensor):
                topk_cases = self.cases[topk_global_indices]  # [B, K, D...] depending on your case shape
                most_activated_cases = topk_cases# .detach()
            else:
                most_activated_cases = None  # keep API; you only use labels/weights downstream

            # Set outputs (detach to avoid keeping big graphs for logging/metrics)
            most_activated_case_labels = topk_labels#.detach()

            # Keep grads; normalize within K
            eps = torch.finfo(top_k_activations.dtype).eps
            w_k = top_k_activations
            sum_k = w_k.sum(dim=1, keepdim=True)

            # handle rare all-zero rows (e.g., after masking)
            all_zero = (sum_k <= eps)
            uniform = torch.full_like(w_k, 1.0 / w_k.size(1))
            w_k_norm = torch.where(all_zero, uniform, w_k / sum_k.clamp_min(eps))

            # If you want one tensor used everywhere:
            most_activated_activations = w_k_norm

            # If you prefer raw for prediction but normalized for locality:
            # most_activated_activations_raw = w_k
            # most_activated_activations_norm = w_k_norm
        # -----------------------------------------------------------------------
        return final_predictions, predicted_solution, pre_adapted_solution, most_activated_cases, most_activated_case_labels, most_activated_activations

def train_model(X_train, y_train, X_val, y_val, feature_extractor, cfg): # , glocal_fw_set_num=glocal_fw_set_num
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
    
    # Build effective feature extractor pipeline (upstream -> optional post-MLP)
    feature_extractor = build_effective_feature_extractor(feature_extractor, cfg)
    if feature_extractor is not None:
        feature_extractor = feature_extractor.to(device)
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
        # pre-train adapter here (switchable via cfg["cdh_aggregate"])
        if cfg.get("nn_cdh_pretrain", False):

            cdh_aggregate = bool(cfg.get("cdh_aggregate", True))  # False=pair pretrain, True=aggregate pretrain

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

            rng = np.random.default_rng(cfg.get("nn_cdh_pair_seed", 0))
            pairs_per_target = cfg.get("nn_cdh_pairs_per_case", 5)

            if not cdh_aggregate:
                # ===== Pair-mode pretrain (existing) =====
                # Build (X_cdh, y_cdh): [context, diff] -> Δy
                X_cdh = np.empty((0, 2 * D), dtype=np.float32)
                Y_cdh = np.empty((0, 1), dtype=np.float32)

                for j in range(N):
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

                # Fit pair net
                adapter.fit_pairs(
                    X_cdh, Y_cdh,
                    device=device,
                )

            else:
                # ===== Aggregate-mode pretrain (pairs, but trains aggregate net) =====
                # Build (X_agg, Y_agg): [diff, y_source] -> Δy
                # This lets aggregate-mode later treat a weighted prototype as a "source case".
                X_agg = np.empty((0, D + 1), dtype=np.float32)  # [diff (D), y_source (1)]
                Y_agg = np.empty((0, 1), dtype=np.float32)      # [Δy]

                for j in range(N):
                    src_indices = rng.choice(N, size=pairs_per_target, replace=False)
                    x_target = Z_np[j:j+1, :]      # [1, D]
                    y_target = y_np[j:j+1, :]      # [1, 1]
                    for i in src_indices:
                        x_source = Z_np[i:i+1, :]  # [1, D]
                        y_source = y_np[i:i+1, :]  # [1, 1]

                        diff = x_target - x_source                        # [1, D]
                        X_one = np.concatenate([diff, y_source], axis=1)  # [1, D+1]
                        Y_one = y_target - y_source                       # [1, 1]

                        X_agg = np.concatenate([X_agg, X_one], axis=0)
                        Y_agg = np.concatenate([Y_agg, Y_one], axis=0)

                # Fit aggregate net (requires adapter.fit_aggregate implemented)
                adapter.fit_aggregate(
                    X_agg, Y_agg,
                    device=device,
                )
    model = NN_KNN_Model(X_train, ys, feature_extractor=feature_extractor, glocal_weightor=glocal_weightor, nn_cdh= adapter, **cfg)
    model.to(device)

    # ---- Optional: freeze feature extractor for early epochs ----
    freeze_epochs = int(cfg.get("freeze_feature_extractor_epochs", 0))
    if freeze_epochs > 0 and model.feature_extractor is not None:
        for p in model.feature_extractor.parameters():
            p.requires_grad_(False)
        print(f"[freeze] feature_extractor frozen for first {freeze_epochs} epochs")

    # Separate parameters for different learning rates
    feature_extractor_params = list(model.feature_extractor.parameters()) if model.feature_extractor is not None else []
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
    base_adapter_lr = adapter_lr

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
    eps_sigma_multiplier = cfg.get("eps_sigma_multiplier", 0.1)
    print(f"Training started for training_epochs epochs with batch size {cfg.get('batch_size')}")

    def _locality_scale(epoch: int, cfg: dict) -> float:
        warm = int(cfg.get("locality_warmup_epochs", 0))
        ramp = int(cfg.get("locality_ramp_epochs", 0))
        if epoch < warm:
            return 0.0
        if ramp <= 0:
            return 1.0
        t = (epoch - warm) / max(1, ramp)
        return float(max(0.0, min(1.0, t)))

    def _scaled_locality_cfg(cfg: dict, scale: float) -> dict:
        # Scale ONLY locality-related lambdas; keep lambda_base as-is.
        if scale >= 1.0:
            return cfg
        cfg2 = dict(cfg)
        for k in ["lambda_kl", "lambda_expdist", "lambda_balance", "lambda_cover", "lambda_pair", "lambda_ent"]:
            cfg2[k] = float(cfg.get(k, 0.0)) * scale
        return cfg2

    ##==========Training loop===========

    # -------------------------
    # Helpers
    # -------------------------
    def _set_requires_grad(params, flag: bool):
        for p in params:
            p.requires_grad_(flag)

    def _freeze_standalone_params(model, flag: bool):
        # Safety: these might be nn.Parameter but not included in your param lists
        if hasattr(model, "biases") and isinstance(model.biases, torch.nn.Parameter):
            model.biases.requires_grad_(flag)
        if hasattr(model, "glocal_weights") and isinstance(model.glocal_weights, torch.nn.Parameter):
            model.glocal_weights.requires_grad_(flag)

    def _validate_one_epoch(epoch_idx: int, stage_tag: str):
        model.eval()
        val_loss_total = 0.0
        final_predictions_list, predicted_solution_list = [], []

        if task != "classification":
            WLD_vals, NA_hits, nDCG_vals = [], [], []
            eps_eval = eps_sigma_multiplier * global_sigma_y

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_final_predictions, batch_predicted_solution, batch_pre_adapted_solution, b_idx, b_labels, b_weights = model(batch_X)

                if task == "classification":
                    base_val = criterion(batch_final_predictions, batch_y)
                    val_loss_batch = base_val
                else:
                    # reuse your existing locality schedule (stage-local epoch index)
                    loc_scale = _locality_scale(epoch_idx, cfg)
                    cfg_loc   = _scaled_locality_cfg(cfg, loc_scale)

                    base_val = criterion(batch_final_predictions.squeeze(1), batch_y.float())

                    if cfg.get("regression_locality", False) and getattr(model, "top_k_mode", False):
                        val_loss_batch, _ = reg_locality_reg_loss(
                            base_val, batch_y, b_labels, b_weights, global_sigma_y, cfg_loc, eps_sigma_multiplier
                        )
                    else:
                        val_loss_batch = base_val

                # case-bias reg (consistent with train; if biases frozen this is constant)
                lambda_case_bias = float(cfg.get("lambda_case_bias", 0.0))
                if lambda_case_bias != 0.0 and hasattr(model, "biases"):
                    bias_l2 = (model.biases ** 2).mean()
                    val_loss_batch = val_loss_batch + lambda_case_bias * bias_l2

                val_loss_total += val_loss_batch.item() * batch_X.size(0)
                final_predictions_list.append(batch_final_predictions)
                predicted_solution_list.append(batch_predicted_solution)

                # neighbor metrics (unchanged)
                # if task != "classification" and getattr(model, "top_k_mode", False):
                #     y_true  = batch_y.float().unsqueeze(1)
                #     abs_d   = torch.abs(b_labels - y_true)

                #     epsw = torch.finfo(b_weights.dtype).eps
                #     w = b_weights.clamp_min(epsw)
                #     w = w / w.sum(dim=1, keepdim=True).clamp_min(epsw)

                #     WLD_vals.append((w * abs_d).sum(dim=1))
                #     K_use = b_weights.size(1)
                #     NA_hits.append((abs_d <= eps_eval).float().mean(dim=1))

                #     gains = torch.exp(-abs_d / (eps_eval + 1e-8))
                #     denom = torch.log2(torch.arange(2, K_use + 2, device=gains.device).float())
                #     dcg   = (gains[:, :K_use] / denom[:K_use]).sum(dim=1)
                #     ideal = torch.sort(gains, dim=1, descending=True).values
                #     idcg  = (ideal[:, :K_use] / denom[:K_use]).sum(dim=1).clamp_min(1e-8)
                #     nDCG_vals.append(dcg / idcg)

        num_val = len(val_loader.dataset) if hasattr(val_loader, "dataset") else len(y_val)
        val_loss = val_loss_total / max(1, num_val)

        # print + metric (kept similar to your current style)
        if task == "classification":
            acc = accuracy_score(y_val.cpu().numpy(), torch.cat(predicted_solution_list).cpu().numpy())
            print(f"[{stage_tag}] Epoch {epoch_idx+1} - Val Acc: {acc:.4f} | Val Loss: {val_loss:.4f}")
            metric = acc
        else:
            predicted_solution = torch.cat(predicted_solution_list, dim=0)
            ss_res = torch.sum((y_val.float() - predicted_solution.float()) ** 2).item()
            ss_tot = torch.sum((y_val.float() - torch.mean(y_val.float())) ** 2).item()
            r2 = 1 - (ss_res / ss_tot)
            print(f"[{stage_tag}] Epoch {epoch_idx+1} - Val R²: {r2:.4f} | Reg Val Loss: {val_loss:.4f}")
            metric = r2

            # if getattr(model, "top_k_mode", False):
            #     WLD  = torch.cat(WLD_vals).mean().item()
            #     NA   = torch.cat(NA_hits).mean().item()
            #     nDCG = torch.cat(nDCG_vals).mean().item()
            #     print(f"[{stage_tag}] Neighbor | WLD: {WLD:.4f} | NA@{K_use}: {NA:.3f} | nDCG: {nDCG:.3f}")

        return val_loss, metric


    # -------------------------
    # Setup + checkpoints
    # -------------------------
    checkpoint_path = cfg.get("checkpoint_path", default_args["checkpoint_path"]) or "nnknn_tmp.pth"
    if checkpoint_path.endswith(".pth"):
        retr_ckpt = checkpoint_path.replace(".pth", "_retr.pth")
        cdh_ckpt  = checkpoint_path.replace(".pth", "_cdh.pth")
    else:
        retr_ckpt = checkpoint_path + "_retr.pth"
        cdh_ckpt  = checkpoint_path + "_cdh.pth"

    patience = cfg.get("patience", default_args.get("patience", 20))
    training_epochs = cfg.get("training_epochs", default_args["training_epochs"])
    eps_sigma_multiplier = cfg.get("eps_sigma_multiplier", 0.1)

    warm_epochs = int(cfg.get("case_norm_warmup_epochs", 0))
    warm_type  = cfg.get("case_norm_warmup_type", "softmax")
    final_type = cfg.get("case_normalizer", "softmax")

    base_adapter_lr = cfg.get("adapter_lr", default_args["adapter_lr"])


    # ==========================================================
    # Stage 1: Retrieval-only (NN-CDH hard disabled)
    # ==========================================================
    print(f"\n[Stage 1] Retrieval-only training for up to {training_epochs} epochs (patience={patience})")

    model.adapt_enabled = False

    # Freeze adapter params
    _set_requires_grad(adapter_params, False)

    # Train retrieval parts
    _set_requires_grad(feature_extractor_params + glocal_weightor_params + case_net_params, True)

    # IMPORTANT: actually freeze feature_extractor at start if using freeze_epochs
    if freeze_epochs > 0 and model.feature_extractor is not None:
        for p in model.feature_extractor.parameters():
            p.requires_grad_(False)

    # Optimizer LR: adapter group -> 0, others unchanged
    # groups: 0=feature_extractor, 1=glocal, 2=adapter, 3=case_net
    if len(optimizer.param_groups) >= 4:
        optimizer.param_groups[0]["lr"] = fearture_extractor_lr
        optimizer.param_groups[1]["lr"] = glocal_weightor_lr
        optimizer.param_groups[2]["lr"] = 0.0
        optimizer.param_groups[3]["lr"] = case_net_lr

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(training_epochs):
        # unfreeze feature extractor at scheduled epoch (stage1 only)
        if freeze_epochs > 0 and model.feature_extractor is not None and epoch == freeze_epochs:
            for p in model.feature_extractor.parameters():
                p.requires_grad_(True)
            print(f"[Stage1/unfreeze] feature_extractor unfrozen at epoch {epoch}")

        # case-normalizer warmup schedule (stage1 only)
        if warm_epochs > 0 and epoch < warm_epochs:
            model.case_normalizer = warm_type
        else:
            model.case_normalizer = final_type

        model.train()

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            final_predictions, predicted_solution, pre_adapted_solution, topk_cases, topk_labels, topk_acts = model(X_batch)

            # ---- reuse your existing loss calculation ----
            if task == "classification":
                base = criterion(final_predictions, y_batch)
                loss = base
            else:
                base = criterion(final_predictions.squeeze(1), y_batch.float())
                if not base.requires_grad:
                    raise RuntimeError("Stage1: loss has no grad. Check requires_grad flags.")
                if cfg.get("regression_locality", False):
                    loc_scale = _locality_scale(epoch, cfg)
                    cfg_loc   = _scaled_locality_cfg(cfg, loc_scale)
                    loss, _ = reg_locality_reg_loss(
                        base, y_batch, topk_labels, topk_acts, global_sigma_y, cfg_loc, eps_sigma_multiplier
                    )
                else:
                    loss = base

            # ---- Case-bias L2 regularizer ----
            lambda_case_bias = float(cfg.get("lambda_case_bias", 0.0))
            if lambda_case_bias != 0.0 and hasattr(model, "biases") and model.biases.requires_grad:
                loss = loss + lambda_case_bias * (model.biases ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        # validate + early stop
        val_loss, _ = _validate_one_epoch(epoch, "Stage1")

        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), retr_ckpt)
            print(f"[Stage1] New best (epoch {epoch+1}) Val Loss: {val_loss:.4f} — saved: {retr_ckpt}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[Stage1] No improv. Best so far: {best_val_loss:.4f} (epoch {best_epoch+1})")
            if patience_counter > patience:
                print("[Stage1] Patience exceeded. Restoring best retrieval model.")
                model.load_state_dict(torch.load(retr_ckpt))
                break

    # Restore best retrieval model before stage 2
    if os.path.exists(retr_ckpt):
        model.load_state_dict(torch.load(retr_ckpt))


    # ==========================================================
    # Stage 2: Adapter-only (NN-CDH enabled; freeze retrieval)
    # ==========================================================
    if adapter is None or len(adapter_params) == 0:
        print("\n[Stage 2] Skipped: adapter not enabled.")
        # final save
        torch.save(model.state_dict(), checkpoint_path)
    else:
        print(f"\n[Stage 2] Adapter-only training for up to {training_epochs} epochs (patience={patience})")

        model.adapt_enabled = True
        model.case_normalizer = final_type  # fixed in stage2

        # Freeze retrieval parts
        _set_requires_grad(feature_extractor_params + glocal_weightor_params + case_net_params, False)
        _freeze_standalone_params(model, False)

        # Train adapter only
        _set_requires_grad(adapter_params, True)

        # Optimizer LR: only adapter group learns
        if len(optimizer.param_groups) >= 4:
            optimizer.param_groups[0]["lr"] = 0.0
            optimizer.param_groups[1]["lr"] = 0.0
            optimizer.param_groups[2]["lr"] = base_adapter_lr
            optimizer.param_groups[3]["lr"] = 0.0

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        for epoch in range(training_epochs):
            model.train()

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                final_predictions, predicted_solution, pre_adapted_solution, topk_cases, topk_labels, topk_acts = model(X_batch)

                # ---- reuse your existing loss calculation (unchanged) ----
                if task == "classification":
                    base = criterion(final_predictions, y_batch)
                    loss = base
                else:
                    base = criterion(final_predictions.squeeze(1), y_batch.float())
                    if not base.requires_grad:
                        raise RuntimeError("Stage2: loss has no grad. Adapter may be frozen or adapt gate not active.")
                    if cfg.get("regression_locality", False):
                        loc_scale = _locality_scale(epoch, cfg)
                        cfg_loc   = _scaled_locality_cfg(cfg, loc_scale)
                        loss, _ = reg_locality_reg_loss(
                            base, y_batch, topk_labels, topk_acts, global_sigma_y, cfg_loc, eps_sigma_multiplier
                        )
                    else:
                        loss = base

                # biases are frozen now; keep for consistency (constant term)
                lambda_case_bias = float(cfg.get("lambda_case_bias", 0.0))
                if lambda_case_bias != 0.0 and hasattr(model, "biases"):
                    loss = loss + lambda_case_bias * (model.biases ** 2).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            # validate + early stop
            val_loss, _ = _validate_one_epoch(epoch, "Stage2")

            if epoch == 0 or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), cdh_ckpt)
                print(f"[Stage2] New best (epoch {epoch+1}) Val Loss: {val_loss:.4f} — saved: {cdh_ckpt}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"[Stage2] No improv. Best so far: {best_val_loss:.4f} (epoch {best_epoch+1})")
                if patience_counter > patience:
                    print("[Stage2] Patience exceeded. Restoring best adapter model.")
                    model.load_state_dict(torch.load(cdh_ckpt))
                    break

        if os.path.exists(cdh_ckpt):
            model.load_state_dict(torch.load(cdh_ckpt))

        # final save
        torch.save(model.state_dict(), checkpoint_path)


    print("Training completed. Best Acc or R2: ", metric_for_model_select)

    print("Final global feature weights:", glocal_weightor.get_feature_weights_display(detach=True))
    return metric_for_model_select, glocal_weightor, model

def reg_locality_reg_loss(
    base_loss: torch.Tensor,
    y_batch: torch.Tensor,
    topk_labels: torch.Tensor,
    topk_activations: torch.Tensor,
    global_sigma_y: torch.Tensor,
    cfg: dict,
    eps_sigma_multiplier: float,
) -> tuple[torch.Tensor, dict]:
    """
    Returns (total_loss, components) where components has:
      kl, expdist, cover, pairwise, entropy, eps_used
    """
    # Shapes
    y_true  = y_batch.float().unsqueeze(1)           # [B,1]
    y_cases = topk_labels                            # [B,K]
    # normalize once
    eps = torch.finfo(topk_activations.dtype).eps
    w = topk_activations.clamp_min(eps)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)  # [B,K]

    sigma_y = global_sigma_y                          # [1,1]
    eps_abs = eps_sigma_multiplier * sigma_y          # [1,1]

    # mix
    lambda_base  = cfg.get("lambda_base", 1.0)
    lambda_kl = cfg.get("lambda_kl", 0.0)            # default 0 if you don't want KL
    lambda_exp   = cfg.get("lambda_expdist", 0.1)
    lambda_cover = cfg.get("lambda_cover", 0.0)
    lambda_pair  = cfg.get("lambda_pair", 0.0)
    lambda_ent   = cfg.get("lambda_ent", 0.0)
    lambda_balance  = cfg.get("lambda_balance",  0.0)    # tune this
    lambda_case_bias = cfg.get("lambda_case_bias", 0.0)  # default 0 if you don't want case bias reg

    # ----- (optional) KL(p || w) -----
    if lambda_kl != 0.0:
        p_unnorm = torch.exp(-torch.abs(y_cases - y_true) / sigma_y)     # [B,K]
        p = p_unnorm / (p_unnorm.sum(dim=1, keepdim=True) + 1e-12)
        kl_loss = (p * (p.add(1e-12).log() - w.log())).sum(dim=1).mean()
    else:
        kl_loss = torch.tensor(0.0, device=w.device)

    # ----- label-aware locality (no KL) -----
    # scaled distance
    dist = torch.abs(y_cases - y_true) / sigma_y     # [B,K]
    tau   = eps_abs / sigma_y                        # == eps_sigma_multiplier

    # hinge on excess distance outside eps (alpha=1 → L1 hinge; 2 → squared)
    alpha = cfg.get("locality_alpha", 2.0)
    if lambda_exp != 0.0:
        excess = torch.clamp(dist - tau, min=0.0)
        expdist_loss = (w * (excess ** alpha)).sum(dim=1).mean()
    else:
        expdist_loss = torch.tensor(0.0, device=w.device)

    # 2) NEW: signed-bias regularizer
    signed = (y_cases - y_true) / sigma_y            # [B, K]
    mean_signed = (w * signed).sum(dim=1)            # [B]
    #Symmetric: penalizes both positive and negative bias
    balance_dist_loss = (mean_signed ** 2).mean()         # scalar, we use power 2 here to prevent negative number
    
    # a special expdist loss for showing locality regularization effect
    # similar to expdist_loss but without hinge at tau
    locality_loss = (w * dist ** alpha).sum(dim=1).mean()


    # coverage: want ≥ target_mass inside eps
    if lambda_cover != 0.0:
        near_mask = (dist <= (tau + 1e-12)).float()
        near_mass = (w * near_mask).sum(dim=1)           # [B]
        target_mass = cfg.get("locality_target_mass", 0.5)
        cover_loss = torch.relu(target_mass - near_mass).mean()
    else:
        cover_loss = torch.tensor(0.0, device=w.device)

    if lambda_pair != 0.0:
        pos = (torch.abs(y_cases - y_true) <= eps_abs).float()                   # [B,K]
        neg = (torch.abs(y_cases - y_true) >= (2.0 * eps_abs)).float()           # [B,K]
        if (pos.sum() > 0) and (neg.sum() > 0):
            wp = w.unsqueeze(2)                          # [B,K,1]
            wn = w.unsqueeze(1)                          # [B,1,K]
            mask = pos.unsqueeze(2) * neg.unsqueeze(1)   # [B,K,K]
            margin = cfg.get("pairwise_margin", 0.05)
            pairwise = torch.clamp(margin - (wp - wn), min=0.0) * mask
            pairwise_loss = pairwise.sum() / (mask.sum() + 1e-8)
        else:
            pairwise_loss = torch.tensor(0.0, device=w.device)
    else:
        pairwise_loss = torch.tensor(0.0, device=w.device)

    # entropy (ADD to penalize entropy → sparser weights)
    if lambda_ent != 0.0:
        entropy = -(w * (w.add(1e-12)).log()).sum(dim=1).mean()
    else:
        entropy = torch.tensor(0.0, device=w.device)
    total = (
        lambda_base  * base_loss
        + lambda_kl    * kl_loss
        + lambda_exp   * expdist_loss
        + lambda_cover * cover_loss
        + lambda_pair  * pairwise_loss
        + lambda_ent   * entropy
        + lambda_balance * balance_dist_loss
    )

    comps = dict(
        base = base_loss.detach(),
        locality_loss = locality_loss.detach(),
        kl=kl_loss.detach(),
        expdist=expdist_loss.detach(),
        cover=cover_loss.detach(),
        pairwise=pairwise_loss.detach(),
        entropy=entropy.detach(),
        eps_used=eps_abs.detach(),
    )
    return total, comps


def cross_validate(Xs, ys, feature_extractor, cfg, k_folds=10):
    """
    Perform k-fold cross-validation using the train_model function.

    Args:
        Xs: Feature tensor.
        ys: Labels.
        cfg: Configuration object.
        k_folds: Number of cross-validation folds.

    Returns:
        best_accuracies: List of best accuracies for each fold.
    """
    k_fold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_accuracies = []
    last_model = None
    for train_index, test_index in k_fold.split(Xs):
        X_train, X_test = Xs[train_index], Xs[test_index]
        y_train, y_test = ys[train_index], ys[test_index]

        best_accuracy, _, last_model = train_model(X_train, y_train, X_test, y_test, feature_extractor, cfg)
        best_accuracies.append(best_accuracy)
        # break

    print("Cross-validation results:", best_accuracies)
    print(f"Average accuracy: {np.mean(best_accuracies):.3f}")
    print(f"Standard deviation: {np.std(best_accuracies):.3f}")
    print(f"{np.mean(best_accuracies):.3f} ({np.std(best_accuracies):.3f})")
    return best_accuracies, last_model


def train_with_given_split(X_train, y_train, X_test, y_test, feature_extractor, cfg):
    """
    Train NN-kNN directly with a provided train/test split.

    Args:
        X_train: Training feature tensor.
        y_train: Training labels.
        X_test: Test feature tensor.
        y_test: Test labels.
        cfg: Configuration object.
    """
    best_accuracy, glocal_weightor, model = train_model(X_train, y_train, X_test, y_test, feature_extractor, cfg)
    print(f"Accuracy on provided split: {best_accuracy:.3f}")
    #print("Final global feature weights:", glocal_weightor.feature_weights)
    return best_accuracy, glocal_weightor, model
