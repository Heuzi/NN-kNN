import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def Zebra():
    # Set seed for reproducibility
    np.random.seed(42)

    # Number of points per class
    num_points = 110

    # Generate alternating classes along the X-axis
    x = np.linspace(0, 100, (int) (num_points/10))
    #repeat X 10 times
    x = np.repeat(x, 10)
    y = np.random.rand(num_points)*100
    labels = np.zeros(num_points)

    # Assign alternating classes
    labels[x%20 == 0] = 0
    labels[x%20 != 0] = 1
    
    Xs = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)
    ys = torch.tensor(labels, dtype=torch.long)
    return Xs, ys

def Zebra_Special():
    # Set seed for reproducibility
    np.random.seed(42)

    # Number of points per class
    num_points = 110

    # Generate alternating classes along the X-axis
    x1 = np.linspace(0, 100, (int) (num_points/10))
    #repeat X 10 times
    x1 = np.repeat(x1, 10)
    y1 = np.random.rand(num_points)*100
    labels1 = np.zeros(num_points)

    # Assign alternating classes
    labels1[x1%20 == 0] = 0
    labels1[x1%20 != 0] = 1

    # Generate alternating classes along the X-axis
    y2 = np.linspace(0, 100, (int) (num_points/10))
    #repeat X 10 times
    y2 = np.repeat(y2, 10)
    x2 = 100 + np.random.rand(num_points)*100
    labels2 = np.zeros(num_points)

    # Assign alternating classes
    labels2[y2%20 == 0] = 0
    labels2[y2%20 != 0] = 1

    x = np.concatenate((x1,x2))
    y = np.concatenate((y1,y2))
    labels = np.concatenate((labels1,labels2))

    Xs = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)
    ys = torch.tensor(labels, dtype=torch.long)
    return Xs, ys

def BAL():
    with open("data/balance-scale.data","r") as filef:
        bal_file = filef.readlines()
        Xs = []
        ys = []
        for line in bal_file:
            Xs.append([int(line[2]),int(line[4]),int(line[6]),int(line[8])])
            if line[0] == 'L':
                ys.append(0)
            elif line[0] == 'B':
                ys.append(1)
            elif line[0] == 'R':
                ys.append(2)
        Xs = torch.tensor(Xs).float()
        ys = torch.tensor(ys)
    return Xs, ys

def Digits(is_norm=True):
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    Xs = digits['data']
    ys = digits['target']

    if is_norm:
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(Xs)

    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys)
    return Xs, ys

def Iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    Xs = iris['data']
    ys = iris['target']
    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys)
    return Xs, ys

def Wine():
    from sklearn.datasets import load_wine
    wine = load_wine()
    Xs = wine['data']
    ys = wine['target']
    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys)
    return Xs, ys

def Breast_Cancer(is_norm=True):
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    scaler = MinMaxScaler()
    Xs = cancer.data
    ys = cancer.target
    if is_norm:
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(Xs)

    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys)
    return Xs, ys

DATATYPES = {
    'zebra':Zebra,
    'zebra_special': Zebra_Special,
    'bal': BAL,
    'digits': Digits,
    'iris': Iris,
    'wine': Wine,
    'breast_cancer': Breast_Cancer
}
def Cls_small_data(dataset):
    Xs, ys = DATATYPES[dataset]()
    return Xs, ys
