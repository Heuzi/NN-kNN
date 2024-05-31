import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import FeatureActivationLayer, CaseActivationLayer, TopCaseLayer

class ClassActivationLayer(nn.Module):
  '''
    measures the activation of a class given some case activations

    input:
      m case_activations
    output:
      l class_activations
  '''
  def __init__(self, num_cases, case_labels):
    super().__init__()
    self.constraints = []
    self.case_labels = case_labels
    self.num_classes = torch.unique(case_labels).shape[0]

    self.ca_weight = nn.Parameter(torch.ones((num_cases, self.num_classes))) ## should I use randn here?
    self.bias = nn.Parameter(torch.ones(self.num_classes))

    # create a mask
    self.mask = torch.zeros_like(self.ca_weight)
    # print("self.ca_weight.shape", self.ca_weight.shape)

  def update_mask(self):
    for i in range(self.mask.shape[0]):
      for j in range(self.mask.shape[1]):
        if(j == self.case_labels[i]):
          if("case_pos_class" in self.constraints):
            self.mask[i][j] = torch.ones(1)
        else: # j != case_labels[i]
          if("case_no_contribute_to_wrong_class" in self.constraints):
            self.mask[i][j] = torch.zeros(1)

  ##NOTE:: the following three constrain uses for loop, may not be the most efficient
  def case_class_constrain_v1(self):
    '''
      ensures that each case only positively activates their correct class label.
    '''
    self.constraints.append("case_pos_class")
    self.update_mask()


  ##NOTE:: if used, this should be called before the constrain_v1()
  def cases_share_weight_on_same_class(self):
    '''
      all cases of the same class share one weight for that class.
    '''
    self.constraints.append("case_share_weight_on_same_class")


  def case_class_constrain_v2(self):
    '''
      ensures that each case does not contribute to incorrect classes.
    '''
    self.constraints.append("case_no_contribute_to_wrong_class")
    self.update_mask()


  def get_constrained_weight(self):
    constrained_weight = self.mask * torch.relu(self.ca_weight)
    return constrained_weight

  def forward(self, ca):
    constrained_weight = self.get_constrained_weight()
    return torch.matmul(ca, constrained_weight) + self.bias
  

class NN_k_NN(nn.Module):
  def __init__(self, 
               cases, 
               case_labels,
               ca_weight_sharing,
               top_case_enabled, 
               top_k,
               discount,
               hidden_layers=None,
               device=None,
               ):
    super().__init__()
    self.cases = cases
    num_cases = cases.shape[0]

    num_features = cases.shape[1]

    self.device = device

    if not (hidden_layers is None):
        num_features = hidden_layers(cases).shape[1]
    
    self.fa_layer = FeatureActivationLayer(num_features, self.cases, hidden_layers = hidden_layers).to(self.device)

    self.ca_layer = CaseActivationLayer(num_features, num_cases, discount, 
                                        weight_sharing= ca_weight_sharing).to(self.device)
    self.top_case_enabled = top_case_enabled
    self.selection_layer = TopCaseLayer(top_k).to(device)
    self.class_layer = ClassActivationLayer(num_cases, case_labels).to(self.device)
    self.class_layer.mask = self.class_layer.mask.to(self.device)

    self.class_layer.case_class_constrain_v1()
    self.class_layer.case_class_constrain_v2()

  def forward(self, query):
    feature_activations = self.fa_layer(query, self.cases)
    
    case_activations = self.ca_layer(feature_activations)

    if self.top_case_enabled:
      case_activations = self.selection_layer(case_activations)

    if self.training:
      dquery = query.unsqueeze(1)
      dcases = self.cases.unsqueeze(0)
      difference = torch.sum(dquery - dcases, dim=-1)
      masks = torch.where(difference == 0, 0., 1.)
      case_activations = case_activations * masks

    class_ativations = self.class_layer(case_activations)

    output = class_ativations
    _, predicted_class = torch.max(output, 1)

    return feature_activations, case_activations, output, predicted_class