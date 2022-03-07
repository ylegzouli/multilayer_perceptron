#%%
import pickle
from utils import load_data
from MLP import MLP

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def create_model(X, y):
    mlp = MLP() 
    pipe = Pipeline([('MLP', mlp)])
    model = pipe.fit(X, y)
    return model

#%%

if __name__ == "__main__":
    
    X, y = load_data()

    model = create_model(X, y)

    scores = cross_val_score(model, X, y, cv=3)

    print(scores)
    
    pickle.dump(model, open('model.save', 'wb'))

#%%