import numpy as np
import pandas as pd

DATA_PATH = 'data.csv'

def load_data(data_path=DATA_PATH):
    # open csv
    df = pd.read_csv(data_path, header=None)
    # get y data
    y = np.array([df[1]])
    y = np.where(y == 'M', 1, 0).T
    # get X data
    X = np.array(df.iloc[:,2:])
    # normalize X data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def split_train_test(X, y, test_size=0.3):
    idx = X.shape[0] - int(X.shape[0] // (1 / test_size))
    X_train, X_test = X[:idx], X[idx:]
    y_train, y_test = y[:idx], y[idx:]
    return X_train.T, X_test.T, y_train.T, y_test.T

class StandardScaler():
    def __init__(self):
        self.mean= 0
        self.std = 0
    
    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        return self
    
    def fit_transform(self, X):
        self.fit(X)
        return (X - self.mean) / self.std

# def dataset_minmax(dataset):
#     minmax = list()
#     for i in range(len(dataset[0])):
#         col_values = [row[i] for row in dataset]
#         value_min = min(col_values)
#         value_max = max(col_values)
#         minmax.append([value_min, value_max])
#     return minmax

# # Rescale dataset columns to the range 0-1
# def normalize_dataset(dataset, minmax):
#     for row in dataset:
#         for i in range(len(row)):
#             row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
def softmax(X):
    max = np.max(X,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(X - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def score(target, output, score_func='rmse'):
    if score_func == 'accuracy':
        output = np.where(output > 0.5, 1, 0)
        res = (output == target).mean()
    if score_func == 'entropy':
        log_likelihood = (-np.log(output.T[range(target.T.shape[0]), target.T]))
        res = (np.sum(log_likelihood) / target.T.shape[0] / 10000 )
    if score_func == 'rmse':
        res = np.average((target - output) ** 2)
    return res
 