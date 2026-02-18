import os
os.environ["WANDB_API_KEY"] = "9c667feed64719efa17839b4211fc9b1768fb629"
# os.environ['KERAS_HOME'] = '/l/vision/manectric_ssd2/minun/ssd2/yw173/nnKnn-P/data'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Union, Dict, Tuple
from numpy.typing import ArrayLike
from tqdm import tqdm
import numpy as np
import random
import argparse
import json
import math
import uuid

from sklearn.metrics import accuracy_score

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def MNIST():
    from keras import datasets
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data(
        path=os.path.join(os.getcwd(), 'data/mnist.npz')
    )

    X_train = torch.tensor(X_train).float().unsqueeze(1) / 255.
    X_test = torch.tensor(X_test).float().unsqueeze(1) / 255.
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()

    return X_train, y_train, X_test, y_test

def CIFAR10():
    import pickle
    print('Loading CIFAR10!')
    def load_batch(file):
        with open(file, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        data = d[b'data']
        labels = d[b'labels']
        data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # Normalize here if needed
        return data, labels

    def load_cifar10_from(path):
        Xs, ys = [], []
        for i in range(1, 6):
            d, l = load_batch(os.path.join(path, f'data_batch_{i}'))
            Xs.append(d)
            ys.extend(l)
        X_train = np.concatenate(Xs)
        y_train = np.array(ys)
        X_test, y_test = load_batch(os.path.join(path, 'test_batch'))
        return (X_train, y_train), (X_test, np.array(y_test))

    (X_train, y_train), (X_test, y_test) = load_cifar10_from('./datasets/cifar-10-batches-py')

    X_train = torch.tensor(X_train).float()             # shape: [N, 3, 32, 32]
    y_train = torch.tensor(y_train).long()              # shape: [N]
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).long()

    return X_train, y_train, X_test, y_test

DATATYPES = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
}

def Cls_medium_data(dataset):
    X_train, y_train, X_test, y_test = DATATYPES[dataset]()
    return X_train, y_train, X_test, y_test

    
def wrap_1dtensor(tensor):
  if tensor.dim() == 0:
    tensor = tensor.unsqueeze(0)
  return tensor

def wrap_inttensor(input):
  if isinstance(input, list):
    input = torch.tensor(input, dtype=int)
  return input

class WAE_Encoder(nn.Module):
    def __init__(self, channel_size=1, last_spatial=4, latent_dim=20):
        """
        A deterministic encoder for a Wasserstein Autoencoder (WAE).
        It maps an input image directly to a latent vector z.
        """
        super().__init__()
        # Encoder: (B, 1, 28, 28) -> (B, 128, 4, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_size, 32, 3, 2, 1),   # (B, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # (B, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), # (B, 128, 4, 4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        
        self.fc_z = nn.Linear(128 * last_spatial * last_spatial, latent_dim)

    def forward(self, x):
      features = self.encoder(x)
      features = torch.flatten(features, start_dim=1)
      z = self.fc_z(features)
      return z
  
class NN_k_NN_GMM(nn.Module):
  def __init__(self, 
               cases,
               case_labels,
               casebase_size:int=50,
               selected_indices:Optional[Union[Tuple, ArrayLike]]=None,
               top_case_enabled:bool=True, 
               top_k:int=5,
               last_spatial:int=4,
               latent_size:int=20,
               encoder_type:str='vaee', # vae, nn, vaee,
               kernel_type:str='rbf',
               rbf_sigma:int=20,
               dataset_name:str='mnist',
               **kwargs,
               ):
    """
    Args:
      cases: (N, F) N:total size of the cases; F:feature size
      cases_labels: [N] a list of the labels associated with the cases
    """

    super().__init__()
    
    self.k = top_k
    self.encoder_type = encoder_type
    self.encoder = WAE_Encoder(channel_size=1 if dataset_name == 'mnist' else 3, last_spatial=last_spatial, latent_dim=latent_size)

    cls_labels = torch.unique(case_labels)
    self.num_classes = len(cls_labels)
    self.empty_nlabel = self.num_classes+1
    
    self.casebase_size = casebase_size

    self.case_fea_size = cases.shape[1:]
    self.latent_size = latent_size
    self.kernel_type = kernel_type
    self.rbf_sigma = rbf_sigma

    casebase = torch.zeros(
      (self.casebase_size, *self.case_fea_size), 
      requires_grad=False
    )
    casebase_labels = torch.full(
      (self.casebase_size,), 
      self.empty_nlabel, 
      dtype=torch.long, 
      requires_grad=False
    )

    # random select the same number of cases from each class
    indices_inclass = []
    size_inclasses = []
    for idx in range(self.num_classes):
      indices = torch.nonzero(case_labels == idx, as_tuple=False).squeeze()
      indices_inclass.append(indices)
      size_inclasses.append(len(indices))
      
    size_inclasses = torch.tensor(size_inclasses, dtype=torch.long, requires_grad=False)

    if self.casebase_size % self.num_classes == 0 and torch.min(size_inclasses).item() > self.casebase_size // self.num_classes:
      actual_size_inclasses = torch.tensor([self.casebase_size // self.num_classes for i in range(self.num_classes)], dtype=torch.long, requires_grad=False)
    else:
      actual_size_inclasses = self._distributeCounts(self.casebase_size, size_inclasses)
      
    if sum(actual_size_inclasses) < len(actual_size_inclasses):
       actual_size_inclasses = self._shuffleDistribute(actual_size_inclasses)

    print(actual_size_inclasses)

    assert torch.sum(actual_size_inclasses) == self.casebase_size
    
    # intialize the casebase
    num_cases = 0
    if selected_indices is None: # no specific selection
      # randomly select the same number of cases from each class
      selected_indices = []
      for idx in range(self.num_classes):
        actual_size_inclass = actual_size_inclasses[idx].item()
        indices = indices_inclass[idx]

        rselected_indices = indices[torch.randperm(len(indices), requires_grad=False)[:actual_size_inclass]]
        selected_indices.extend(rselected_indices)

        casebase[num_cases:num_cases+actual_size_inclass] = cases[rselected_indices]
        casebase_labels[num_cases:num_cases+actual_size_inclass] = case_labels[rselected_indices]
        num_cases += actual_size_inclass
      
      selected_indices = torch.LongTensor(selected_indices)
    else:
      if isinstance(selected_indices, tuple):
        selected_indices, fixed_indices = selected_indices
      else:
        fixed_indices = None
        
      selected_indices = wrap_inttensor(selected_indices)
      fixed_indices = wrap_inttensor(fixed_indices)
        
      if len(selected_indices) < self.casebase_size:
        selected_labels = case_labels[selected_indices]
        new_selected_indices = []
        for idx in range(self.num_classes):
          al_select_indices_inclass = wrap_1dtensor(torch.nonzero(selected_labels == idx, as_tuple=False).squeeze())
          
          al_select = selected_labels[al_select_indices_inclass]
          actual_size_inclass = actual_size_inclasses[idx].item() - len(al_select)
          accumulated_actual_size_inclasses = torch.cumsum(actual_size_inclasses, dim=0)
          accumulated_actual_size_inclasses = torch.cat((torch.tensor([0]), accumulated_actual_size_inclasses), dim=0)

          rselected_indices = torch.zeros((actual_size_inclasses[idx], ), dtype=int)
          indices = indices_inclass[idx]
          _al_select_indices = selected_indices[al_select_indices_inclass]
          _indices = indices[~torch.isin(indices, _al_select_indices)]
          _temp_indices_all = _indices[torch.randperm(len(_indices), requires_grad=False)[:actual_size_inclass]]
          
          if fixed_indices is not None:
            _filtered_fixed_indices = wrap_1dtensor(torch.nonzero(
                (fixed_indices >= accumulated_actual_size_inclasses[idx]) & 
                (fixed_indices < accumulated_actual_size_inclasses[idx + 1]), 
                as_tuple=False
            ).squeeze())
            
            fixed_indices_inclass = fixed_indices[_filtered_fixed_indices] - accumulated_actual_size_inclasses[idx]
            rselected_indices[fixed_indices_inclass] = _al_select_indices
            _temp_indices = torch.arange(actual_size_inclasses[idx])
            rest_indices_inclass = _temp_indices[~torch.isin(_temp_indices, fixed_indices_inclass)]
            rselected_indices[rest_indices_inclass] = _temp_indices_all
          else:
            rselected_indices[:len(al_select)] = _al_select_indices
            rselected_indices[len(al_select):] = _temp_indices_all

          casebase[accumulated_actual_size_inclasses[idx]:accumulated_actual_size_inclasses[idx+1]] = cases[rselected_indices]
          casebase_labels[accumulated_actual_size_inclasses[idx]:accumulated_actual_size_inclasses[idx+1]] = case_labels[rselected_indices]
          num_cases += actual_size_inclasses[idx]
          
          new_selected_indices.extend(rselected_indices)
        
        selected_indices = torch.LongTensor(new_selected_indices)
      
      elif len(selected_indices) == self.casebase_size:
        casebase = cases[selected_indices]
        casebase_labels = case_labels[selected_indices]
        num_cases = len(selected_indices)
      else:
        raise ValueError(f"Please check your input selection")
      
    self.selected_indices = selected_indices

    self.register_buffer("mask", torch.full((self.casebase_size, self.num_classes), -1.0))
    self.register_buffer("casebase", casebase)
    self.register_buffer("casebase_labels", casebase_labels)

    # top k
    self.top_case_enabled = top_case_enabled

    # class activation layer
    self.class_activate_weight = nn.Parameter(torch.ones((self.casebase_size, self.num_classes)))
    self.class_activate_bias = nn.Parameter(torch.ones(self.num_classes))

    # feature activation layer
    self.L = nn.Parameter(torch.eye(latent_size))
    # self.logit_alpha = nn.Parameter(torch.tensor(1))
    # self.register_buffer("logit_alpha", torch.tensor(1.0))
    self.temperature = nn.Parameter(torch.tensor(1.0))

    init_scale = 3.0 # 3.0
    mu_init = torch.randn(self.num_classes, self.latent_size) * init_scale 
    mu_init = F.normalize(mu_init, dim=1) * init_scale

    logvar_init = torch.full((self.num_classes, self.latent_size), -2.0)  # exp(-2)=0.135
    logvar_init += 0.1 * torch.randn_like(logvar_init)

    self.mu = torch.nn.Parameter(mu_init)
    self.logvar = torch.nn.Parameter(logvar_init)

    self.register_buffer("log_2pi", torch.tensor(math.log(2 * math.pi)))

    # create a mask
    self._updateMask()
    
    total_params = sum(p.numel() for p in self.parameters())
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

  @torch.no_grad()
  def _shuffleDistribute(self, actual_size_inclasses):
    indices = torch.randperm(actual_size_inclasses.size(0))
    shuffled_A = actual_size_inclasses[indices]
    return shuffled_A

  @torch.no_grad()
  def _distributeCounts(self, N: int, lengths):
    total_length = lengths.sum().item()

    if N > total_length:
        raise ValueError(f"N={N} is larger than the total number of elements {total_length}.")
    if N < 0:
        raise ValueError("N must be non-negative.")

    ratios = (N * lengths.float()) / float(total_length)

    base_counts = torch.floor(ratios).long()
    base_counts = torch.min(base_counts, lengths)

    current_sum = base_counts.sum().item()
    leftover = N - current_sum
    if leftover < 0:
        raise RuntimeError(
            "Somehow we assigned more than N due to a rounding/clamping issue. "
            "Check your logic or inputs."
        )
    if leftover == 0:
        return base_counts

    fractional_parts = ratios - base_counts.float()
    indices_not_full = (base_counts < lengths).nonzero().flatten()  # only indices that can still accept increments
    
    not_full_info = [(idx.item(), fractional_parts[idx].item()) for idx in indices_not_full]
    
    not_full_info.sort(key=lambda x: x[1], reverse=True)

    i = 0
    while leftover > 0 and not_full_info:
        idx, _ = not_full_info[i]
        if base_counts[idx] < lengths[idx]:
            base_counts[idx] += 1
            leftover -= 1
        i += 1
        if i == len(not_full_info):
            not_full_info = [
                (ix, frac) for (ix, frac) in not_full_info 
                if base_counts[ix] < lengths[ix]
            ]
            i = 0
    return base_counts
  
  @torch.no_grad()
  def reinitialize(self,):
    self.feature_active_weight[:] = 1

    # case activation layer
    self.case_active_weight[:] = 1

    # class activation layer
    self.class_activate_weight[:] = 1
    self.class_activate_bias = nn.Parameter(torch.ones(self.num_classes))

    self.mask = torch.zeros_like(self.class_activate_weight)
    self._updateMask()
  
  @torch.no_grad()
  def remove_cases(self, case_indices:list):
    remain_indices = []
    for i in range(self.casebase_size):
      if i not in case_indices:
        remain_indices.append(i)
    remain_indices = torch.LongTensor(remain_indices) 
    
    self.casebase_size -= len(case_indices)
      
    new_casebase = torch.zeros((self.casebase_size, *self.case_fea_size), requires_grad=False)
    new_casebase_labels = torch.full((self.casebase_size,), self.empty_nlabel, dtype=torch.long, requires_grad=False)
    new_casebase = self.casebase[remain_indices]
    new_casebase_labels = self.casebase_labels[remain_indices]
    self.casebase = new_casebase
    self.casebase_labels = new_casebase_labels
    
    self.class_activate_weight = nn.Parameter(self.class_activate_weight[remain_indices])
    self.mask = torch.zeros_like(self.class_activate_weight)
    self._updateMask()
    
  @torch.no_grad()
  def rebuild_casebase(self, num_cases, cases, case_labels):
    self.casebase_size = num_cases
    self.casebase = torch.zeros((self.casebase_size, *self.case_fea_size), requires_grad=False)
    self.casebase_labels = torch.full((self.casebase_size,), self.empty_nlabel, dtype=torch.long, requires_grad=False)
    
    # random select the same number of cases from each class
    indices_inclass = []
    actual_size_inclasses = []
    for idx in range(self.num_classes):
      indices = torch.nonzero(case_labels == idx, as_tuple=False).squeeze()
      indices_inclass.append(indices)
      actual_size_inclasses.append(len(indices))

    actual_size_inclasses = torch.tensor(actual_size_inclasses, dtype=torch.long, requires_grad=False)
    actual_size_inclasses = self._distributeCounts(self.casebase_size, actual_size_inclasses)

    print(actual_size_inclasses)

    assert torch.sum(actual_size_inclasses) == self.casebase_size

    num_cases = 0
    for idx in range(self.num_classes):
      actual_size_inclass = actual_size_inclasses[idx].item()
      indices = indices_inclass[idx]

      rselected_indices = indices[torch.randperm(len(indices), requires_grad=False)[:actual_size_inclass]]

      self.casebase[num_cases:num_cases+actual_size_inclass] = cases[rselected_indices]
      self.casebase_labels[num_cases:num_cases+actual_size_inclass] = case_labels[rselected_indices]
      num_cases += actual_size_inclass

    self.mask = torch.zeros((self.casebase_size, self.num_classes))
    self._updateMask()
  
  @torch.no_grad()
  def _updateMask(self):
    """
    Idea: a case should also be able to contribute to classify a query case from another class in a negative way 
    """
    self.mask[:, :] = -1.
    for index in range(len(self.casebase)):
        self.mask[index][self.casebase_labels[index]] = 1.

  def _mahalanobis_distance(self, z_q, z_cb):
    """
    z_q: (B, F)
    z_cb: (M, F)
    """
    L_tril = torch.tril(self.L)
    M_mat  = L_tril.T @ L_tril

    delta   = z_q[:,None,:] - z_cb[None,:,:]
    proj    = F.linear(delta, M_mat)
    d2_mahal = (delta * proj).sum(-1)

    # cidx     = self.casebase_labels
    # mu_j     = self.mu[cidx].detach()
    # logvar_j = torch.clamp(self.logvar[cidx], min=np.log(1e-6)).detach()
    # var_j    = torch.exp(logvar_j)

    # diff    = z_q[:,None,:] - mu_j[None,:,:]
    # nll     = 0.5*(diff.pow(2)/var_j[None,:,:] + logvar_j[None,:,:] + self.log_2pi)
    # d2_nll  = nll.sum(-1)

    # alpha = torch.sigmoid(self.logit_alpha)

    # hybrid = alpha * d2_mahal + (1 - alpha) * d2_nll

    # return F.softmax(-hybrid / (self.temperature + 1e-6), dim=-1)
    return F.softmax(-d2_mahal / (self.temperature + 1e-6), dim=-1)

  
  def topK(self, input):
    if self.training:
      return input
    '''
      input: m case activations
      output: m case activations, the top k activations are kept and others are zeroed out
    '''
    vals, idx = torch.topk(input, self.k)
    output = torch.zeros_like(input).scatter_(1, idx, vals)
    return output
  
  def _compute_kernel(self,
                     x1,
                     x2,
                     latent_var=2.0):
    x1 = x1.unsqueeze(1)  # (N, 1, D)
    x2 = x2.unsqueeze(0)  # (1, N, D)

    if self.kernel_type == 'rbf':
        # sigma = 2. * self.latent_size * latent_var
        # sigma = self._median_bandwidth(torch.cat([x1.squeeze(1), x2.squeeze(0)], dim=0))
        dist = ((x1 - x2) ** 2).sum(-1)
        return torch.exp(-dist / self.rbf_sigma)

    elif self.kernel_type == 'imq':
        C = 2. * self.latent_size * latent_var
        dist = ((x1 - x2) ** 2).sum(-1)
        return C / (C + dist)

    else:
        raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
  def _wae_mmd_mog_regularization(self,
                                  z, 
                                  labels,
                                  reg_weight=100.0):
    mmd = 0.0
    valid_classes = 0

    for k in range(self.num_classes):
        z_k = z[labels == k]
        if z_k.size(0) < 2:
            continue

        mu_k = self.mu[k]
        std_k = torch.exp(0.5 * self.logvar[k])

        prior_z_k = torch.randn_like(z_k) * std_k + mu_k

        k_zz = self._compute_kernel(z_k, z_k)
        k_pp = self._compute_kernel(prior_z_k, prior_z_k)
        k_zp = self._compute_kernel(z_k, prior_z_k)

        mmd_k = k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()
        mmd += mmd_k
        valid_classes += 1

    if valid_classes > 0:
        mmd /= valid_classes
    return reg_weight * mmd
  
  def classActivation(self, input):
    constrained_weight = self.mask * torch.relu(self.class_activate_weight) # shape (num_cases, num_classes)
    return torch.matmul(input, constrained_weight) + self.class_activate_bias
  
  # def _centroid_separation_loss(self):
  #   mu = self.mu  # (K, F)
    
  #   dist_sq = torch.cdist(mu, mu, p=2).pow(2)
    
  #   dist_sq = dist_sq + torch.eye(mu.size(0), device=mu.device) * 1e10
    
  #   min_dist_sq = torch.min(dist_sq) + 1e-8 
  #   loss = 1.0 / min_dist_sq
  #   return loss

  def _centroid_separation_loss(self, num_samples=128):
    loss = 0.0
    num_pairs = 0
    for k in range(self.num_classes):
        for j in range(k + 1, self.num_classes):
            std_k = torch.exp(0.5 * self.logvar[k])
            prior_z_k = torch.randn(num_samples, self.latent_size, device=self.mu.device) * std_k + self.mu[k]

            std_j = torch.exp(0.5 * self.logvar[j])
            prior_z_j = torch.randn(num_samples, self.latent_size, device=self.mu.device) * std_j + self.mu[j]

            k_kk = self._compute_kernel(prior_z_k, prior_z_k)
            k_jj = self._compute_kernel(prior_z_j, prior_z_j)
            k_kj = self._compute_kernel(prior_z_k, prior_z_j)
            mmd_kj = k_kk.mean() + k_jj.mean() - 2 * k_kj.mean()

            loss += 1.0 / (mmd_kj + 1e-8)
            num_pairs += 1

    if num_pairs > 0:
        return loss / num_pairs
    return 0.0

  # def _centroid_separation_loss(self, separation_loss_weight=0.01):
  #       mu = self.mu         # shape: (num_classes, latent_size)
  #       logvar = self.logvar # shape: (num_classes, latent_size)
        
  #       logvar = torch.clamp(logvar, -10, 10)
  #       var = torch.exp(logvar)

  #       total_kl_loss = 0.0
  #       num_pairs = 0

  #       for i in range(self.num_classes):
  #           for j in range(i + 1, self.num_classes):
  #               mu_i, var_i, logvar_i = mu[i], var[i], logvar[i]
  #               mu_j, var_j, logvar_j = mu[j], var[j], logvar[j]

  #               # 0.5 * [ tr(Σ_j^-1 * Σ_i) + (μ_j-μ_i)^T Σ_j^-1 (μ_j-μ_i) - F + log(det(Σ_j)/det(Σ_i)) ]
                
  #               # Term 1: tr(Σ_j^-1 * Σ_i) -> sum(var_i / var_j)
  #               term1_ij = torch.sum(var_i / var_j)
  #               # Term 2: (μ_j-μ_i)^T Σ_j^-1 (μ_j-μ_i) -> sum( (μ_j-μ_i)^2 / var_j )
  #               term2_ij = torch.sum((mu_j - mu_i)**2 / var_j)
  #               # Term 3: log(det(Σ_j)/det(Σ_i)) -> sum(logvar_j) - sum(logvar_i)
  #               term3_ij = torch.sum(logvar_j) - torch.sum(logvar_i)
  #               # Latent dimension F
  #               F = self.latent_size

  #               kl_ij = 0.5 * (term1_ij + term2_ij - F + term3_ij)
                
  #               term1_ji = torch.sum(var_j / var_i)
  #               term2_ji = torch.sum((mu_i - mu_j)**2 / var_i)
  #               term3_ji = torch.sum(logvar_i) - torch.sum(logvar_j)

  #               kl_ji = 0.5 * (term1_ji + term2_ji - F + term3_ji)
                
  #               symmetric_kl = kl_ij + kl_ji
                
  #               pair_loss = 1.0 / (symmetric_kl + 1e-8)
  #               total_kl_loss += pair_loss
  #               num_pairs += 1

  #       if num_pairs > 0:
  #           return separation_loss_weight * (total_kl_loss / num_pairs)
  #       return torch.tensor(0.0, device=mu.device)


  def _logvar_regularization(self, min_logvar=-2.0, penalty_weight=1):
    penalty = F.relu(min_logvar - self.logvar).mean()
    return penalty_weight * penalty

  def compute_loss(self,
                   logit,
                   labels,
                   z_q):

    cls_loss = F.cross_entropy(logit, labels)

    mmd_loss = self._wae_mmd_mog_regularization(z_q, labels)
    cen_loss = self._centroid_separation_loss()
    logvar_loss = self._logvar_regularization()

    return cls_loss, mmd_loss, cen_loss, logvar_loss

  def forward(self, query):
    z_q = self.encoder(query)
    z_cb = self.encoder(self.casebase)

    case_activations = self._mahalanobis_distance(z_q, z_cb)

    if self.top_case_enabled:
      case_activations = self.topK(case_activations)

    if self.training:
        dquery = query.unsqueeze(1)            # (B,1,C,H,W)
        dcases = self.casebase.unsqueeze(0)     # (1,M,C,H,W)

        diff = (dquery - dcases).abs().flatten(start_dim=2)    # (B, M, C*H*W)
        max_diff, _ = diff.max(dim=2)                          # (B, M)

        match = max_diff < 1e-6                                # (B, M)
        case_activations = case_activations.masked_fill(match, 0.0)

    output = self.classActivation(case_activations)    
    return output, case_activations, z_q
    
    
def eval_gmm_model(model,
                   test_loader):
    model.eval()
    predicted_classes = []
    ground_truth = []
    total_cls_loss, total_mmd_loss, total_cen_loss, total_logvar_loss = 0., 0., 0., 0.
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), desc="Eval", ncols=80)
        for batch_idx, (query, label) in enumerate(test_loader):
            query = query.cuda()
            output, _, z_q = model(query)
            cls_loss, mmd_loss, cen_loss, logvar_loss = model.compute_loss(output, label.cuda(), z_q)
            
            total_cls_loss += cls_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_cen_loss += cen_loss.item()
            total_logvar_loss += logvar_loss.item()
        
            predicted_class = torch.argmax(F.softmax(output, dim=-1), dim=-1)
            predicted_classes.extend(predicted_class.cpu().numpy())
            ground_truth.extend(label.tolist())
            
            pbar.update(1)  
        pbar.close()
        
    accuracy = accuracy_score(ground_truth, predicted_classes)
    total_cls_loss = total_cls_loss/(batch_idx+1)
    total_mmd_loss = total_mmd_loss/(batch_idx+1)
    total_cen_loss = total_cen_loss/(batch_idx+1)
    total_logvar_loss = total_logvar_loss/(batch_idx+1)

    return accuracy, total_cls_loss, total_mmd_loss, total_cen_loss, total_logvar_loss


def format_weight(w):
    return f"{int(w)}" if 1 <= w < 100 else f"{w:.0e}"

def train_gmm_model(model,
                    train_loader,
                    lr,
                    weight_mmd,
                    weight_cen,
                    weight_logvar,
                    rbf_sigma,
                    dataset_name,
                    casebase_size,
                    batch_size,
                    latent_size,
                    max_epochs,
                    enable_wandb,
                    **kwargs):
    
    short_uuid = uuid.uuid4().hex[:6]
    args.uuid = short_uuid
    print(args)
    if enable_wandb:
        import wandb
        
        if args.note:
           wandb_run_name = f"gmm3_{args.note}_{dataset_name}_{casebase_size}_{format_weight(weight_mmd)}_c{format_weight(weight_cen)}_lo{format_weight(weight_logvar)}_{rbf_sigma}_b{batch_size}_l{latent_size}"
        else:
           wandb_run_name = f"gmm3_{dataset_name}_{casebase_size}_{format_weight(weight_mmd)}_c{format_weight(weight_cen)}_lo{format_weight(weight_logvar)}_{rbf_sigma}_b{batch_size}_l{latent_size}"
        
        wandb_run = wandb.init(
            entity="yuwang1-indiana-university",
            project="VaeKNN",
            group="gmm3",
            name=wandb_run_name,
            notes="This experiment is conducted to see the performance.",
            config=vars(args)
        )
        # wandb.watch(model, log='all', log_freq=100)

    optim = torch.optim.Adam(model.parameters(), lr=lr) 
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.95)
    best_val_acc = -1

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.
        total_cls_loss, total_mmd_loss, total_cen_loss, total_logvar_loss = 0, 0, 0, 0

        # regular training
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", ncols=160)
        
        for batch_idx, (X_train_batch, y_train_batch) in enumerate(train_loader): 
            X_train_batch = X_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)

            optim.zero_grad()
            output, _, z_q = model(X_train_batch)
            cls_loss, mmd_loss, cen_loss, logvar_loss = model.compute_loss(output, y_train_batch, z_q)
            
            loss = cls_loss + weight_mmd * mmd_loss + weight_cen * cen_loss + weight_logvar * logvar_loss
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_cen_loss += cen_loss.item()
            total_logvar_loss += logvar_loss.item()

            pbar.update(1)
            pbar.set_postfix({'Loss': total_loss/(batch_idx+1),
                              'Cls': cls_loss.item(),
                              'MMD': mmd_loss.item(), 
                              'Cen': cen_loss.item(),
                              'Logvar' : logvar_loss.item()})
        
        pbar.close()
        scheduler.step()

        total_loss = total_loss/(batch_idx+1)
        total_cls_loss = total_cls_loss/(batch_idx+1)
        total_mmd_loss = total_mmd_loss/(batch_idx+1)
        total_cen_loss = total_cen_loss/(batch_idx+1)
        total_logvar_loss = total_logvar_loss/(batch_idx+1)
        
        val_accuracy, val_cls_loss, val_mmd_loss, val_cen_loss, val_logvar_loss = eval_gmm_model(model, test_loader)
        print(f'Eval [{epoch + 1}], ACC: {val_accuracy:.4f}, Val Loss: {val_cls_loss:.4f}, MMD: {val_mmd_loss:.4f}, Cen: {val_cen_loss:.4f}, LogVar: {val_logvar_loss:.4f}')

        if enable_wandb:
            wandb_run.log({"train/loss_cls": total_cls_loss,
                        "train/loss_mmd": total_mmd_loss,
                        "train/loss_cen": total_cen_loss,
                        "train/loss_logvar": total_logvar_loss,
                        "val/acc": val_accuracy,
                        "val/loss_cls": val_cls_loss,
                        "val/loss_mmd": val_mmd_loss,
                        "val/loss_cen": val_cen_loss,
                        "val/loss_logvar": val_logvar_loss,
                        "parameters/temperature": model.temperature.item(),
                        # "parameters/logit_alpha": model.logit_alpha.item(),
                        "gradient/mu_hist": wandb.Histogram(model.mu.grad.cpu().numpy()),
                        "gradient/ogvar_hist": wandb.Histogram(model.logvar.grad.cpu().numpy()),
                        "parameters/mu_hist": wandb.Histogram(model.mu.detach().cpu().numpy()),
                        "parameters/logvar_hist": wandb.Histogram(model.logvar.detach().cpu().numpy()),
                        })

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            # save the model
            if args.save_ckpt:
                torch.save({'acc': best_val_acc,
                          'epoch': epoch,
                          'args': vars(args),
                          'model': model.state_dict()}, f"checkpoints/{short_uuid}_gmm3_{dataset_name}_{casebase_size}_{latent_size}.pt")

    print(f'Best Validation Accuracy: {best_val_acc:.4f}')


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 python train_GMM3.py --dataset_name 'cifar10' --casebase_size 1000 --batch_size 32 --max_epochs 30 --latent_size 20 --lr 1e-3 --weight_mmd 0.1 --weight_cen 1 --weight_logvar 0.1 --rbf_sigma 80 --note "steplr" --save_ckpt --enable_wandb
    """
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--dataset_name', type=str, default='mnist', help='Dataset to train')
    parser.add_argument('--ca_weight_sharing', action='store_true', help='Enable CA weight sharing')
    parser.add_argument('--top_case_enabled', action='store_true', help='Enable top case')
    parser.add_argument('--class_weight_sharing', action='store_true', help='Enable class weight sharing')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--discount', type=int, default=2, help='Discount factor')
    parser.add_argument('--train_split_ratio', type=float, default=0.1, help='Train split ratio')
    parser.add_argument('--temp_add_size', type=float, default=0.01, help='Temporary addition size')
    parser.add_argument('--training_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_mmd', type=float, default=0.1, help='Weight for MMD loss')
    parser.add_argument('--weight_cen', type=float, default=0.1, help='Weight for Separation loss')
    parser.add_argument('--weight_logvar', type=float, default=0.1, help='Weight for Logvar Penalty loss')
    parser.add_argument('--kernel_type', type=str, default='rbf', help='Kernel for MMD')
    parser.add_argument('--rbf_sigma', type=int, default=20, help='Kernel for MMD')
    parser.add_argument('--note', type=str, default=None, help='string argument')


    parser.add_argument('--casebase_size', type=int, default=1000, help='The size of the case base')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=30, help='Batch size')
    parser.add_argument('--encoder_type', type=str, default='gmm', help='Encoder Type')
    parser.add_argument('--top_k', type=int, default=5, help='Top K value')
    parser.add_argument('--latent_size', type=int, default=20, help='Top K value')
    parser.add_argument('--enable_wandb', action='store_true', help='Enable the wandb log')
    parser.add_argument('--save_ckpt', action='store_true', help='Enable the model saving')

    args = parser.parse_args()

    
    with open('selected_indices.json', 'r') as f:
       selected_indices = json.load(f)[str(args.casebase_size)]

    X_train, y_train, X_test, y_test = Cls_medium_data(args.dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    test_set = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=30)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=30)

    model = NN_k_NN_GMM(X_train,
                        y_train,
                        selected_indices=selected_indices,
                        **vars(args)
                        ).to(device)
    
    train_gmm_model(model, train_loader, **vars(args))