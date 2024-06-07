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

# Function to convert categorical values to numeric and handle irregular values
def convert_to_numeric(value):
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
    dfs = [pd.read_csv(file_path) for file_path in three_files]
    # Concatenate the DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Display columns to identify irrelevant features
    print("Columns in the dataset:", df.columns)
    #https://static1.squarespace.com/static/5e8769b34812765cff8111f7/t/5e99d902ca4a0277b8b5fb51/1587140880354/COVID-19+Tracking+Survey+Questionnaire+041720.pdf
    

    # Identify columns with more than 100 missing values
    missing_values = combined_df.isnull().sum()
    columns_to_remove = missing_values[missing_values > 100].index.tolist()

    # Drop the identified columns
    combined_df.drop(columns=columns_to_remove, inplace=True)


    # Define irrelevant features based on questionnaire sections
    irrelevant_features = [
        'CONSENT','SU_ID','P_PANEL','NATIONAL_WEIGHT','REGION_WEIGHT',
        'NATIONAL_WEIGHT_POP','REGION_WEIGHT_POP','NAT_WGT_COMB_POP','REG_WGT_COMB_POP',
        #droping columns with lots of missing (already done above)
        # 'P_OCCUPY2',
        #dropping predictive labels
        'SOC5A','SOC5B','SOC5C','SOC5D','SOC5E'
    ]
    # Identify and convert columns with categorical answers
    for column in combined_df.columns:
        if combined_df[column].dtype == 'object':
            combined_df[column] = combined_df[column].apply(convert_to_numeric)

    # Drop irrelevant columns   
    combined_df.drop(columns=irrelevant_features, inplace=True, errors='ignore')
    # Drop rows with any missing values
    combined_df = combined_df.dropna()

    y = combined_df['SOC5A']
    X = combined_df.drop(columns=['SOC5A'])

    
    # random_state = 13
    # balancing, currently not enabled
    # from imblearn.over_sampling import SMOTE
    # oversample = SMOTE(random_state=random_state, k_neighbors=3)
    # X, y = oversample.fit_resample(X, y)
    Xs = torch.tensor(X.values).float()
    ys = torch.tensor(y.values).long()
    return Xs, ys
    

DATATYPES = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'covid_anxious':covid_anxious
}
def Cls_medium_data(dataset):
    X_train, y_train, X_test, y_test = DATATYPES[dataset]()
    return X_train, y_train, X_test, y_test
