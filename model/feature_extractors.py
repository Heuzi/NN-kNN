# prompt: a classifier with conv layers for MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust input size based on image dimensions
        self.fc2 = nn.Linear(128, 10)  # Output size is 10 for 10 digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# prompt: a classifier with conv layers for cifar10
class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjust input size based on image dimensions and pooling
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8) # Adjust input size based on image dimensions and pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_feature_extractor_from(model):
    """
    Extracts the feature extraction part of a given model by removing the last layer.
    Assumes the last layer is a fully connected (linear) layer.
    """
    feature_extractor = torch.nn.Sequential(
      model.conv1,
      nn.ReLU(),   # Add ReLU here
      model.pool,
      model.conv2,
      nn.ReLU(),   # Add ReLU here
      model.pool,
      torch.nn.Flatten(),
      model.fc1
    )
    feature_extractor.feature_dim = model.fc1.out_features
    return feature_extractor