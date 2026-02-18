import torch
import numpy as np
from urllib import request
import datasets.cls_small_data
from imblearn.over_sampling import SMOTE
import pandas as pd

def airfoil():
    file_path = "/content/drive/MyDrive/NN-kNN/datasets/airfoil_self_noise.dat"
    # file_path = 'G:/My Drive/NN-kNN/datasets/airfoil_self_noise.dat'
    df = pd.read_csv(file_path, sep="\t", header=None, engine="python")

    # feature columns vs target column
    feature_cols = [0, 1, 2, 3, 4]
    target_col = 5

    df_clean = df.dropna()

    # Convert to torch tensors
    X = torch.tensor(df_clean[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df_clean[target_col].values, dtype=torch.float32).unsqueeze(1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X,y

def student_performance():
    # file_path = '/content/drive/MyDrive/Personal Work/datasets/student-por.csv'
    file_path = '/content/drive/MyDrive/NN-kNN/datasets/student-por.csv'

    df = pd.read_csv(file_path, sep=';')

    # Target
    y = df['G3']

    # Drop target
    X = df.drop(columns=['G3'])

    # One-hot encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # convert columns to float
    X = X.astype(float)

    # Convert to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    X = X_tensor
    y = y_tensor

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y

def yacht():
    file_path = "/content/drive/MyDrive/NN-kNN/datasets/yacht_hydrodynamics.data"
    # file_path = 'G:/My Drive/NN-kNN/datasets/yacht_hydrodynamics.data'
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    X = df.drop(columns=[6])
    y = df[6]

    X = X.astype(float)
    y = y.astype(float)

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    return X, y

def energy_efficiency():

    file_path = "/content/drive/MyDrive/NN-kNN/datasets/ENB2012_data.csv"

    df = pd.read_csv(file_path)

    X = df.iloc[:, 0:8]
    y = df.iloc[:, 8]

    X = X.astype(float)
    y = y.astype(float)

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    return X, y

def car():
    # file_path = "/content/drive/MyDrive/Personal Work/cars_cleaned_Above600kDeleted_NoDuplicate.csv"
    file_path = "/content/drive/MyDrive/NN-kNN/datasets/cars_cleaned_Above600kDeleted_NoDuplicate.csv"
    # file_path = 'G:/My Drive/NN-kNN/datasets/cars_cleaned_Above600kDeleted_NoDuplicate.csv'
    df = pd.read_csv(file_path)# the features
    feature_cols = ['Year', 'Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg', 'Popularity']
    target_col = 'MSRP' # predicting the retail price
    df_filtered = df[feature_cols + [target_col]].dropna()     # drop missing values

    X = torch.tensor(df_filtered[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df_filtered[target_col].values, dtype=torch.float32).unsqueeze(1)
    return X, y

def psych_depression_physical_symptons_reg():
    #From Zach Wilkerson, ICCBR challenge.
    #"dataset/Dataset_MO_ENG.csv"
    # df = pd.read_csv("/content/drive/Othercomputers/My MacBook Pro/GitHub/NN-kNN/dataset/Dataset_MO_ENG.csv")
    df = pd.read_csv("G:/My Drive/NN-kNN/datasets/Dataset_MO_ENG.csv")
    ## eliminating physical-related questions
    df = df.drop(df.columns[102:-1], axis=1)
    ## Creating classes 0-> Low risk, 1->Medium Risk, 2->High risk
    # dic = { 1: 0 , 2: 0, 3:1, 4:2, 5:2}
    # df['Target'] = df['Target'].map(dic)
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X = df[train_cols]
    
    print(list(X.columns))

    y = df[label]
    target_names=["Low","Medium","High"]
    #balancing the data set
    random_state = 13
    oversample = SMOTE(random_state=random_state, k_neighbors=3)
    X, y = oversample.fit_resample(X, y)
    Xs = torch.tensor(X.values).float()
    ys = torch.tensor(y.values).long()
    ys = ys.float()
    return Xs, ys

def California_Housing():
    from sklearn.datasets import fetch_california_housing

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

    df = pd.read_csv("datasets/bodyfat.csv")

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

def Ziweifaces():
    Xs = np.load("part_features.npy")
    ys = np.load("part_targets.npy")
    #These two files are in the nn-Knn folder.
    Xs = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32) 
    return Xs, ys

def covid_anxious_reg():
    Xs, ys = cls_small_data.covid_anxious()
    ys = torch.tensor(ys, dtype=torch.float32)
    return Xs, ys
def covid_depressed_reg():
    Xs, ys = cls_small_data.covid_depressed()
    ys = torch.tensor(ys, dtype=torch.float32)
    return Xs, ys
def covid_lonely_reg():
    Xs, ys = cls_small_data.covid_lonely()
    ys = torch.tensor(ys, dtype=torch.float32)
    return Xs, ys
def covid_hopeless_reg():
    Xs, ys = cls_small_data.covid_hopeless()
    ys = torch.tensor(ys, dtype=torch.float32)
    return Xs, ys
def covid_physical_reg():
    Xs, ys = cls_small_data.covid_physical()
    ys = torch.tensor(ys, dtype=torch.float32)
    return Xs, ys

# def standardize_tensor(input_tensor):
#     mean = input_tensor.mean()
#     std = input_tensor.std()
#     standardized_tensor = (input_tensor - mean) / std
#     return standardized_tensor

def standardize_tensor(x, dim=0, eps=1e-8, mean=None, std=None, return_stats=False):
    """
    Per-dimension standardization by default (dim=0 for [N,D]).
    If mean/std provided, applies them (useful for val/test).
    """
    if mean is None:
        mean = x.mean(dim=dim, keepdim=True)
    if std is None:
        std = x.std(dim=dim, keepdim=True)
    std = std.clamp_min(eps)
    xz = (x - mean) / std
    if return_stats:
        return xz, mean, std
    return xz

DATATYPES = {
    'student_performance': student_performance,
    'airfoil': airfoil,
    'car': car,
    'yacht': yacht,
    'energy_efficiency': energy_efficiency,
    'psych_depression_physical_symptons_reg':psych_depression_physical_symptons_reg,
    'califonia_housing':California_Housing,
    'abalone': Abalone,
    'diabets': Diabetes,
    'body_fat': Body_Fat,
    'ziweifaces': Ziweifaces,
    'covid_anxious_reg': covid_anxious_reg,
    'covid_depressed_reg':covid_depressed_reg,
    'covid_lonely_reg':covid_lonely_reg,
    'covid_hopeless_reg':covid_hopeless_reg,
    'covid_physical_reg':covid_physical_reg
}
def Reg_data(dataset):
    
    Xs, ys = DATATYPES[dataset]()
    ##IMPORTANT: This standardization leads to data leakage, so we comment it out for now.
    ## Standardization now happens after train-test split in nnknn_sample_regression.ipynb
    ## MOREVOER, this is not per-feature standardization, but standardization over the entire tensor.
    # Xs = standardize_tensor(Xs)
    # ys = standardize_tensor(ys)
    return Xs, ys