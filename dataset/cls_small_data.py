import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from google.colab import drive
import pandas as pd

def psych_depression_physical_symptons():
    #From Zach Wilkerson, ICCBR challenge.
    #"dataset/Dataset_MO_ENG.csv"
    df = pd.read_csv("/content/drive/Othercomputers/My MacBook Pro/GitHub/NN-kNN/dataset/Dataset_MO_ENG.csv")
    ## eliminating physical-related questions
    df = df.drop(df.columns[102:-1], axis=1)
    ## Creating classes 0-> Low risk, 1->Medium Risk, 2->High risk
    dic = { 1: 0 , 2: 0, 3:1, 4:2, 5:2}
    df['Target'] = df['Target'].map(dic)
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X = df[train_cols]
    y = df[label]
    target_names=["Low","Medium","High"]
    random_state = 13
    from imblearn.over_sampling import SMOTE
    oversample = SMOTE(random_state=random_state, k_neighbors=3)
    X, y = oversample.fit_resample(X, y)
    Xs = torch.tensor(X.values).float()
    ys = torch.tensor(y.values).long()
    return Xs, ys

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

import re

# Function to convert categorical values to numeric and handle irregular values
def convert_to_numeric(value):
    # Check if the value is already a number
    if isinstance(value, (int, float)):
        return value

    # Check if the value is a string representation of a number
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                pass
    # Income range mappings
    # income_mapping = {
    #     "Less than $5,000": 1,
    #     "$5,000 to $9,999": 2,
    #     "$10,000 to $14,999": 3,
    #     "$15,000 to $19,999": 4,
    #     "$20,000 to $24,999": 5,
    #     "$25,000 to $29,999": 6,
    #     "$30,000 to $34,999": 7,
    #     "$35,000 to $39,999": 8,
    #     "$40,000 to $49,999": 9,
    #     "$50,000 to $59,999": 10,
    #     "$60,000 to $74,999": 11,
    #     "$75,000 to $84,999": 12,
    #     "$85,000 to $99,999": 13,
    #     "$100,000 to $124,999": 14,
    #     "$125,000 to $149,999": 15,
    #     "$150,000 to $174,999": 16,
    #     "$175,000 to $199,999": 17,
    #     "$200,000 or more": 18
    # }
    income_mapping = {
        "Under $10,000": 1,
        "$10,000 to under $20,000": 2,
        "$20,000 to under $30,000": 3,
        "$30,000 to under $40,000": 4,
        "$40,000 to under $50,000": 5,
        "$50,000 to under $75,000": 6,
        "$75,000 to under $100,000": 7,
        "$100,000 to under $150,000": 8,
        "$150,000 or more": 9
    }
    
    # Convert income ranges
    if value in income_mapping:
        return income_mapping[value]

    match = re.match(r'\((\d+)\)', str(value))
    if match:
        return int(match.group(1))
    if value in ["(77) Not sure", "(98) SKIPPED ON WEB", "(99) REFUSED", "(88) Removed for disclosure risk"]:
        return np.nan
    return value

# https://www.covid-impact.org/about-the-survey-questionnaire
# Following three data sets are from the same files, but using different features as predictor.
def covid_anxious():
    three_files = ["/content/drive/Othercomputers/My MacBook Pro/GitHub/NN-kNN/dataset/COVID_W1.csv",
    "/content/drive/Othercomputers/My MacBook Pro/GitHub/NN-kNN/dataset/COVID_W2.csv",
    "/content/drive/Othercomputers/My MacBook Pro/GitHub/NN-kNN/dataset/COVID_W3.csv"]
    
    # Read each CSV file into a DataFrame
    dfs = [pd.read_csv(file_path,dtype=str) for file_path in three_files]
    # Concatenate the DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Display columns to identify irrelevant features
    print("Columns in the dataset:", combined_df.columns)
    #https://static1.squarespace.com/static/5e8769b34812765cff8111f7/t/5e99d902ca4a0277b8b5fb51/1587140880354/COVID-19+Tracking+Survey+Questionnaire+041720.pdf
    
    # This is not good, some answers do not show if a pre-req question is answered differently
    # Identify columns with more than 100 missing values
    # missing_values = combined_df.isnull().sum()
    # columns_to_remove = missing_values[missing_values > 100].index.tolist()

    # Define irrelevant features based on questionnaire sections
    irrelevant_features = [
        'SU_ID','P_PANEL',
        'NATIONAL_WEIGHT','REGION_WEIGHT','NATIONAL_WEIGHT_POP',
        'REGION_WEIGHT_POP','NAT_WGT_COMB_POP','REG_WGT_COMB_POP',
        #droping columns with lots of missing (already done above)
        'MAIL50','RACE2_BANNER',
        #dropping predictive labels
        # 'SOC5A',
        'SOC5B','SOC5C','SOC5D','SOC5E'
    ]
    # Drop irrelevant columns   
    combined_df.drop(columns=irrelevant_features, inplace=True, errors='coerce')

    # Identify and convert columns with categorical answers
    for column in combined_df.columns:
        if combined_df[column].dtype == 'object':
            combined_df[column] = combined_df[column].apply(convert_to_numeric)

    # Drop rows with any missing values
    # combined_df = combined_df.dropna()
    combined_df = combined_df.apply(pd.to_numeric, errors='coerce')
    # Fill missing values with 0 (or another number if more appropriate)
    combined_df.fillna(0, inplace=True)
    
    y = combined_df['SOC5A']
    X = combined_df.drop(columns=['SOC5A'])
    
    # Filter out rows where the y value is not between 1 and 4
    valid_indices = y.isin([1, 2, 3, 4])
    X = X[valid_indices]
    y = y[valid_indices]
    #Having labels in the range [1 2 3 4] would cause an "IndexError" because the function anticipates indices in the range [0 1 2 3].
    y = y - 1
    # random_state = 13
    # balancing, currently not enabled
    # from imblearn.over_sampling import SMOTE
    # oversample = SMOTE(random_state=random_state, k_neighbors=3)
    # X, y = oversample.fit_resample(X, y)
    # print(X.values[0])
    Xs = torch.tensor(X.values).float()
    ys = torch.tensor(y.values).long()

    # print(Xs[0])
    return Xs, ys
    
DATATYPES = {
    'psych_depression_physical_symptons':psych_depression_physical_symptons,
    'zebra':Zebra,
    'zebra_special': Zebra_Special,
    'bal': BAL,
    'digits': Digits,
    'iris': Iris,
    'wine': Wine,
    'breast_cancer': Breast_Cancer,
    'covid_anxious':covid_anxious
}
def Cls_small_data(dataset):
    Xs, ys = DATATYPES[dataset]()
    return Xs, ys
