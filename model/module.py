import torch
import torch.nn as nn
import torch.nn.functional as F
  
class FeatureActivationLayer(torch.nn.Module):
  '''
    measures the feature distance/activation between query and all cases
  '''
  def __init__(self, num_features, cases, hidden_layers = None):
    super().__init__()
    #we assume feature weight sharing between segments
    self.feature_matrix = cases
    self.f1weight = torch.nn.Parameter(torch.ones(num_features))

    self.hidden_layers = hidden_layers

  def forward(self, query, cases):
    '''
      input: (m+1) cases * n features; where the first case is the query
      output: m* n feature activations
    '''
    if self.hidden_layers != None:
      query = self.hidden_layers(query)
      cases = self.hidden_layers(cases)

    query = query.unsqueeze(1)
    return (query*self.f1weight - cases*self.f1weight)**2
  
class CaseActivationLayer(torch.nn.Module):
  '''
    measures the activation of a case given some feature activations

    input:
      m* n feature activations
    output:
      m case activations
  '''
  def __init__(self, num_features, num_cases, discount, weight_sharing=False):
    super().__init__()
    self.weight_sharing = weight_sharing
    self.fa_weight = torch.nn.Parameter(torch.ones((num_cases, num_features)))
    if weight_sharing:
      self.fa_weight = torch.nn.Parameter(torch.ones(num_features))

    self.bias = torch.nn.Parameter(torch.ones(num_cases) * num_features / discount)
    # self.bias = torch.nn.Parameter(torch.ones(num_cases) )

  def forward(self,input):
    '''
      input: m*n feature activations
      output: m case activations
    '''
    input = - torch.sum(input * F.leaky_relu(self.fa_weight), dim=2) + self.bias
    input = torch.sigmoid(input)
    return input
  
# prompt: define a class named TopCaseLayer inherited from torch.nn.Module. This class is used to select the top k activations of an input tensor (m case activations), and output a tensor of the same shape but only keep the top k activations, and other tensors zeroed out

class TopCaseLayer(torch.nn.Module):
  def __init__(self, k):
    super().__init__()
    self.k = k
    self.training = True

  def forward(self, input):
    ##no behavior during training because we want to train for all.
    if self.training:
      return input
    '''
      input: m case activations
      output: m case activations, the top k activations are kept and others are zeroed out
    '''
    vals, idx = torch.topk(input, self.k)
    output = torch.zeros_like(input).scatter_(1,idx, vals)

    return output