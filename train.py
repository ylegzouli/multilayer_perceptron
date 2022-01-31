#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'data.csv'

#%%

def load_data(data_path=DATA_PATH):

    # open csv
    df = pd.read_csv(data_path, header=None)
    # print(df.shape)

    # get y data
    y = np.array([df[1]])
    y = np.where(y == 'M', 1, 0).T
    # print(y)

    # get X data
    X = np.array(df.iloc[:,2:])
    # print(X)

    # normalize X data
    scaler = StandardScaler() # with_mean=False ?
    X = scaler.fit_transform(X)
    # print(X)

    # slpit in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # X_train = X_train.T
    # X_test = X_test.T
    # y_train = y_train.reshape((1, len(y_train)))
    # y_test = y_test.reshape((1, len(y_test)))   # Retourner tab ?
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test


#%%
from sklearn.pipeline import Pipeline
from MLP import MLP

def create_model(X_train, y_train):

    # mlp = MLP()
    mlp = MLP(2,[5], 1)
    # mlp = MLP(len(X_train),[5], 2)

    pipe = Pipeline([('MLP', mlp)])
    
    model = pipe.fit(X_train, y_train)
    
    return model

#%%
from sklearn.model_selection import cross_val_score
import pickle

if __name__ == "__main__":
    
    # X_train, X_test, y_train, y_test = load_data()
    from random import random
    X_train = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    y_train = np.array([[i[0] + i[1]] for i in X_train])
    
    model = create_model(X_train, y_train)

    # scores = cross_val_score(model, X_train, y_train, cv=3)
    # print('Score: ', scores.mean())
    
    # Save fit model
    # pickle.dump(model, open('model.save', 'wb'))


#%%