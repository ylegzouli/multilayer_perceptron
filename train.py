#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'data.csv'

#%%

# def softmax(X):
#     max = np.max(X,axis=1,keepdims=True) #returns max of each row and keeps same dims
#     e_x = np.exp(X - max) #subtracts each row with its max value
#     sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
#     f_x = e_x / sum 
#     return f_x

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
    # X = softmax(X)
    # print(X)

    # slpit in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test


#%%
from sklearn.pipeline import Pipeline
from MLP import MLP

def create_model(X_train, y_train):

    # mlp = MLP()
    # mlp = MLP(2,[5], 1)
    mlp = MLP(len(X_train.T),[5, 5], 2)

    pipe = Pipeline([('MLP', mlp)])
    
    model = pipe.fit(X_train, y_train)
    
    return model

#%%
from sklearn.model_selection import cross_val_score
import pickle

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = load_data()
    
    model = create_model(X_train, y_train)

    scores = cross_val_score(model, X_train, y_train, cv=3)
    print('Score: ', scores.mean())
    
    # Save fit model
    # pickle.dump(model, open('model.save', 'wb'))


#%%

# train history:
    # 31-01 0.92129