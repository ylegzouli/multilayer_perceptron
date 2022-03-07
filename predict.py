#%%
import pickle
from utils import load_data, score

DATA_PATH = 'data.csv'

def predict():
    try:
        model = pickle.load(open("model.save", "rb"))
        X, y = load_data(data_path=DATA_PATH)
    except:
        print('Error while loading model or data')
        return
    output = model.predict(X)
    rmse = score(y, output)
    accuracy = score(y, output, score_func='accuracy')
    loss_entropy = score(y, output, score_func='entropy')
    print('Rmse: ', rmse)
    print('Accuracy: ', accuracy)
    print('Loss Entropy: ', loss_entropy)
    return output


#%%

if __name__ == "__main__":
    output = predict()

#%%
