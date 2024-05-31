import torch
from keras import datasets
import os

def MNIST():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data(path=os.path.join(os.getcwd(), 'data/mnist.npz'))
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()
    return X_train, y_train, X_test, y_test


def CIFAR10():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data(path=os.path.join(os.getcwd(), 'data/cifar10.npz'))
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()
    return X_train, y_train, X_test, y_test


DATATYPES = {
    'mnist': MNIST,
    'cifar10': CIFAR10
}
def Cls_medium_data(dataset):
    X_train, y_train, X_test, y_test = DATATYPES[dataset]()
    return X_train, y_train, X_test, y_test
