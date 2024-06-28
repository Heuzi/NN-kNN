import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from google.colab import drive
import pandas as pd
from sklearn.utils import resample
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

def standardize_tensor(input_tensor):
    mean = input_tensor.mean()
    std = input_tensor.std()
    standardized_tensor = (input_tensor - mean) / std
    return standardized_tensor

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
    
    print(list(X.columns))

    y = df[label]
    target_names=["Low","Medium","High"]
    #balancing the data set
    random_state = 13
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
    # NANify the strange values.
    if value in ["(77) Not sure", "(98) SKIPPED ON WEB", "(99) REFUSED", "(88) Removed for disclosure risk"]:
        return np.nan
    match = re.match(r'\((\d+)\)', str(value))
    if match:
        return int(match.group(1))
    return value

def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns)

# https://www.covid-impact.org/about-the-survey-questionnaire
def covid_soc(target):    
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
        'SU_ID','P_PANEL','P_GEO',
        'NATIONAL_WEIGHT','REGION_WEIGHT','NATIONAL_WEIGHT_POP',
        'REGION_WEIGHT_POP','NAT_WGT_COMB_POP','REG_WGT_COMB_POP',
        #droping columns with lots of missing (maybe done above)
        'MAIL50','RACE2_BANNER','P_OCCUPY2','MARITAL','LGBT','PHYS11_TEMP'
        #dropping predictive labels
        ,'SOC5A','SOC5B','SOC5C','SOC5D','SOC5E'
    ]
    if target in irrelevant_features:
        irrelevant_features.remove(target)

    # Drop irrelevant columns   
    combined_df.drop(columns=irrelevant_features, inplace=True, errors='coerce')

    # Identify and convert columns with categorical answers
    for column in combined_df.columns:
        if combined_df[column].dtype == 'object':
            combined_df[column] = combined_df[column].apply(convert_to_numeric)

    # Fill NaN values for specific columns
    combined_df['ECON2'].fillna(value=0, inplace=True)
    combined_df['ECON4'].fillna(value=0, inplace=True)

    # Set display options to print the entire DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(combined_df.head(10))
    # Reset to default display options
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    # Drop rows with any missing values
    combined_df = combined_df.dropna()

    combined_df = combined_df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with 0 (or another number if more appropriate)
    # combined_df.fillna(0, inplace=True)
    
    y = combined_df[target]
    X = combined_df.drop(columns=[target])
    
    # Filter out rows where the y value is not between 1 and 4
    valid_indices = y.isin([1, 2, 3, 4])
    X = X[valid_indices]
    y = y[valid_indices]
    #Having labels in the range [1 2 3 4] would cause an "IndexError" because the function anticipates indices in the range [0 1 2 3].
    y = y - 1

    # Convert nominal features to one-hot encoding
    nominal_columns = []  # 'P_GEO', Replace with your nominal feature columns
    X = one_hot_encode(X, nominal_columns)

    print(X.values[0])
    Xs = torch.tensor(X.values).float()
    ys = torch.tensor(y.values).long()

    random_state = 13
    # # Create the SMOTE+Tomek Links object
    # smote_tomek = SMOTETomek(smote=SMOTE(), tomek=TomekLinks())
    # # Fit and transform the dataset
    # Xs, ys = smote_tomek.fit_resample(Xs, ys)

    # # balancing, currently not enabled
    # oversample = SMOTE(random_state=random_state, k_neighbors=3)
    # Xs, ys = oversample.fit_resample(Xs, ys)

    # Downsampling to match the size of the minority class
    class_counts = np.bincount(ys.numpy())
    min_class_count = np.min(class_counts)
    downsampled_indices = []
    for class_index in np.unique(ys):
        class_indices = np.where(ys == class_index)[0]
        downsampled_class_indices = resample(class_indices, 
                                             replace=False, 
                                             n_samples=min_class_count, 
                                             random_state=random_state)
        downsampled_indices.extend(downsampled_class_indices)
    
    downsampled_indices = np.array(downsampled_indices)
    Xs = Xs[downsampled_indices]
    ys = ys[downsampled_indices]

    np.random.seed(0)
    idx = np.random.permutation(len(Xs))
    Xs = Xs[idx]
    ys = ys[idx] 

    Xs = standardize_tensor(Xs)
    return Xs, ys
def covid_anxious():
    return covid_soc('SOC5A')

def covid_depressed():
    return covid_soc('SOC5B')

def covid_lonely():
    return covid_soc('SOC5C')

def covid_hopeless():
    return covid_soc('SOC5D')

def covid_physical():
    return covid_soc('SOC5E')
    
DATATYPES = {
    'psych_depression_physical_symptons':psych_depression_physical_symptons,
    'zebra':Zebra,
    'zebra_special': Zebra_Special,
    'bal': BAL,
    'digits': Digits,
    'iris': Iris,
    'wine': Wine,
    'breast_cancer': Breast_Cancer,
    'covid_anxious':covid_anxious,
    'covid_depressed':covid_depressed,
    'covid_lonely':covid_lonely,
    'covid_hopeless':covid_hopeless,
    'covid_physical':covid_physical
}
def Cls_small_data(dataset):
    Xs, ys = DATATYPES[dataset]()
    return Xs, ys
