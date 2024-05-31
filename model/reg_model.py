import torch
from model.module import FeatureActivationLayer, CaseActivationLayer, TopCaseLayer

class CustomSoftmaxLayer(torch.nn.Module):
  def __init__(self):
    super(CustomSoftmaxLayer, self).__init__()

  def forward(self, x):
    total = torch.sum(x, dim=1, keepdim=True)
    softmax_output = x / (total + 1e-10)

    ##IMPORTANT, cannot use the softmax version below, because of dividing by 0 problem.

    # # Apply the exponential function to the input tensor
    # exp_x = torch.exp(x)

    # # Apply a mask to set the output to 0 where the input is 0
    # mask = (x != 0).float()  # Create a binary mask (1 where x is not 0, 0 where x is 0)

    # # Apply the mask to the output
    # softmax_output = exp_x * mask

    # # Normalize the output by dividing by the sum along the same axis
    # softmax_output = softmax_output / softmax_output.sum(dim=1, keepdim=True)

    return softmax_output
  
# prompt: Build a regression activation layer just like the class activation layer above, but for the purpose of regression now
## right now this is essentially a linear layer, but somehow it's a lot slower than a linear layer,
class RegressionActivation_1_Layer(torch.nn.Module):
  '''
    measures the activation of a class given some case activations

    input:
      m case_activations
    output:
      1 regression_activations
  '''
  def __init__(self, num_cases, case_labels, weight_sharing=False):
    super().__init__()
    self.constraints = []
    self.num_classes = 1
    self.case_labels = case_labels
    self.ca_weight = torch.nn.Parameter(torch.randn((num_cases, self.num_classes))) ## should I use randn here?
    self.bias = torch.nn.Parameter(torch.randn(self.num_classes))
    print("self.ca_weight.shape", self.ca_weight.shape)
  def forward(self, ca):
    ##IMPORTANT, newly added "* case_labels" here, so that case labels is now considered in
    ##the last layer of NN-k-NN regressor.
    # print("debugg msaafasdf")
    # print(ca.shape)
    # print(self.ca_weight.shape)
    # print(self.case_labels.shape)
    # print((torch.matmul(ca * self.case_labels, self.ca_weight)).shape)
    return torch.matmul(ca, self.ca_weight) + self.bias
  
class RegressionActivation_2_Layer(torch.nn.Module):
  '''
    measures the activation of a class given some case activations

    input:
      m case_activations
    output:
      1 regression_activations
  '''
  def __init__(self, num_cases, case_labels, weight_sharing=False):
    super().__init__()
    self.constraints = []
    self.num_classes = 1
    self.case_labels = case_labels
    self.ca_weight = torch.nn.Parameter(torch.ones((num_cases, self.num_classes))) ## should I use randn here?
    self.bias = torch.nn.Parameter(torch.randn(self.num_classes))
    print("self.ca_weight.shape", self.ca_weight.shape)
  def forward(self, ca):
    return torch.matmul(ca* self.case_labels, self.ca_weight) + self.bias
  
class RegressionActivation_3_Layer(torch.nn.Module):
  '''
    measures the activation of a class given some case activations

    input:
      m case_activations
    output:
      1 regression_activations
  '''
  def __init__(self, num_cases, case_labels, weight_sharing=False):
    super().__init__()
    self.constraints = []
    self.num_classes = 1
    self.case_labels = case_labels
  def forward(self, ca):
    return torch.matmul(ca, self.case_labels) #+ self.bias
  
class RegressionActivation_4_Layer(torch.nn.Module):
  '''
    measures the activation of a class given some case activations

    input:
      m case_activations
    output:
      1 regression_activations
  '''
  def __init__(self, num_cases, case_labels, weight_sharing=False):
    super().__init__()
    self.constraints = []
    self.num_classes = 1
    self.case_labels = case_labels
    self.ca_weight = torch.nn.Parameter(torch.tensor(case_labels).reshape(num_cases, self.num_classes)) ## should I use randn here?
    self.bias = torch.nn.Parameter(torch.randn(self.num_classes))
    print("self.ca_weight.shape", self.ca_weight.shape)
  def forward(self, ca):
    ##IMPORTANT, newly added "* case_labels" here, so that case labels is now considered in
    ##the last layer of NN-k-NN regressor.
    # print("debugg msaafasdf")
    # print(ca.shape)
    # print(self.ca_weight.shape)
    # print(self.case_labels.shape)
    # print((torch.matmul(ca * self.case_labels, self.ca_weight)).shape)
    return torch.matmul(ca, self.ca_weight) + self.bias
  
import torch.nn.functional as F

##IMPORTANT, cannot use top case selection layer here like we do in for classification.
## it has to be either enabled or disabled the whole time.
class NN_k_NN_regression(torch.nn.Module):
  def __init__(self, cases, case_labels,
               fa_weight_sharing_within_segment,
               fa_weight_sharing_between_segment,
               ca_weight_sharing,
               top_case_enabled, top_k,
               class_weight_sharing, hidden_layers= None):
    super().__init__()
    self.cases = cases
    num_cases = cases.shape[0]
    num_features = cases.shape[1]
    
    self.fa_layer = FeatureActivationLayer(num_features, num_cases, self.cases, hidden_layers = hidden_layers)
    self.ca_layer = CaseActivationLayer(num_features, num_cases,
                                        weight_sharing= ca_weight_sharing)
    self.top_case_enabled = top_case_enabled
    self.selection_layer = TopCaseLayer(top_k)

    self.ca_dropout = torch.nn.Dropout(p=0.5)

    self.softmax = CustomSoftmaxLayer()

    self.class_layer = RegressionActivation_3_Layer(num_cases, case_labels,weight_sharing=class_weight_sharing)

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

    case_activations = self.softmax(case_activations)
    values, indices = torch.topk(case_activations, k=self.selection_layer.k, dim=1)

    output = torch.mean(self.class_layer.case_labels[indices], dim=1)
    # print("case_activations" , torch.sum(case_activations, dim= 1))
    predicted_number = self.class_layer(case_activations)
    
    return feature_activations, case_activations, output, predicted_number
