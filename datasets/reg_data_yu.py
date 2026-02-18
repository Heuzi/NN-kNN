import torch
import numpy as np
from urllib import request
from sklearn.datasets import fetch_california_housing
import os
def California_Housing():
    california_housing = fetch_california_housing()
    Xs = california_housing.data
    ys = california_housing.target

    # prompt: Convert Xs and ys to tensors for pytorch

    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32)
    return Xs, ys

def Abalone():
    # Download the Abalone dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    response = request.urlopen(url)
    abalone_data = response.read().decode("utf-8").splitlines()

    # Process and convert the data to PyTorch tensors
    data = [line.strip().split(',') for line in abalone_data]
    X = []
    y = []

    # categories = ['M', 'F', 'I']
    # label_encoder = OneHotEncoder(categories=[categories])
    # label_encoder.fit(data)

    def encode_sex(sex):
        if sex == 'M':
            return [1, 0, 0]
        elif sex == 'F':
            return [0, 1, 0]
        elif sex == 'I':
            return [0, 0, 1]

    for row in data:
        # One-hot encode the 'Sex' feature
        # sex_encoded = label_encoder.transform([[row[0]]])[0]
        sex_encoded = encode_sex(row[0])

        # Convert the row to float and extract the target variable ('Rings')
        X.append(sex_encoded + list(map(float, row[1:-1])))
        y.append(float(row[-1]))

        # # Encode the categorical 'Sex' feature
        # row[0] = label_encoder.transform([row[0]])[0]
        # # Convert the row to float and extract the target variable ('Rings')
        # X.append(list(map(float, row[:-1])))
        # y.append(float(row[-1]))

    Xs = torch.tensor(X, dtype=torch.float32)
    ys = torch.tensor(y, dtype=torch.float32)
    return Xs, ys

def Diabetes():
    # prompt: load the diabetes dataset from sklearn
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    Xs = diabetes.data
    ys = diabetes.target
    # prompt: convert Xs and ys to float tensor
    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32)
    return Xs, ys


def Body_Fat():
    import pandas as pd
    import scipy.stats as stats

    # csv_path = os.path.join(os.path.dirname(__file__), "bodyfat.csv")
    # df = pd.read_csv(csv_path)
    df = pd.read_csv("datasets/bodyfat.csv")   # relative to PROJECT_ROOT (because you chdir there)

    # df = pd.read_csv("/l/vision/gigantamax/yw173/nnKnn-P/data/bodyfat.csv")

    X = df.drop(['BodyFat','Density'],axis=1)
    y = df['Density']
    X['Bmi']=703*X['Weight']/(X['Height']*X['Height'])
    X['ACratio'] = X['Abdomen']/X['Chest']
    X['HTratio'] = X['Hip']/X['Thigh']
    X.drop(['Weight','Height','Abdomen','Chest','Hip','Thigh'],axis=1,inplace=True)
    z = np.abs(stats.zscore(X))

    #only keep rows in dataframe with all z-scores less than absolute value of 3
    X_clean = X[(z<3).all(axis=1)]
    y_clean = y[(z<3).all(axis=1)]
    #find how many rows are left in the dataframe
    Xs = torch.tensor( X_clean.to_numpy(), dtype=torch.float32)
    ys = torch.tensor( y_clean.to_numpy(), dtype=torch.float32)
    return Xs, ys

def Ofaces():
    Xs = np.load("/l/vision/gigantamax/yw173/nnKnn-P/data/part_features.npy")
    ys = np.load("/l/vision/gigantamax/yw173/nnKnn-P/data/part_targets.npy")
    #These two files are in the nn-Knn folder.
    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32) 
    
    #find how many rows are left in the dataframe
    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32)
    return Xs, ys

def Bike_Sharing():
    from ucimlrepo import fetch_ucirepo
    bike_sharing = fetch_ucirepo(id=275) 
    X = bike_sharing.data.features 
    y = bike_sharing.data.targets 
    
    # Drop date string column
    if 'dteday' in X.columns:
        X = X.drop(columns=['dteday'])
    
    # Use only 'cnt' as target
    if 'cnt' in y.columns:
        y = y['cnt']
        
    Xs = torch.tensor(X.values, dtype=torch.float32)
    ys = torch.tensor(y.values, dtype=torch.float32)
    ys = ys.view(-1) # Flatten target
    return Xs, ys

def Wine():
    from ucimlrepo import fetch_ucirepo 
  
    # Fetch dataset (ID 186 is Wine Quality)
    wine_quality = fetch_ucirepo(id=186) 
    
    X = wine_quality.data.features 
    y = wine_quality.data.targets 
    
    if 'color' in X.columns:
        X = X.drop(columns=['color'])
    
    if 'quality' in y.columns:
        y = y['quality']

    Xs = torch.tensor(X.values, dtype=torch.float32)
    ys = torch.tensor(y.values, dtype=torch.float32)
    
    ys = ys.view(-1) 
    
    return Xs, ys

def standardize_tensor(input_tensor):
    mean = input_tensor.mean()
    std = input_tensor.std()
    standardized_tensor = (input_tensor - mean) / std
    return standardized_tensor

DATATYPES = {
    'califonia_housing':California_Housing,
    'abalone': Abalone,
    'diabetes': Diabetes,
    'body_fat': Body_Fat,
    'ofaces': Ofaces,
    'bike_sharing': Bike_Sharing,
    'wine': Wine,
}
def Reg_data(dataset):
    
    Xs, ys = DATATYPES[dataset]()
    Xs = standardize_tensor(Xs)
    ys = standardize_tensor(ys)
    return Xs, ys