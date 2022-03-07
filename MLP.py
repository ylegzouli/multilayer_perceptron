#%%
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import plotly.graph_objects as go
from utils import split_train_test

fig = go.Figure()

#%%

class MLP(BaseEstimator, TransformerMixin):
    def __init__(self):
        nb_inputs = 30
        nb_hidden = 3
        nb_outputs = 1
        np.random.seed(42)
        weight_1 = np.random.randn(nb_hidden, nb_inputs) * 0.01
        weight_2 = np.random.randn(nb_outputs, nb_hidden) * 0.01
        bias_1 = np.zeros((nb_hidden, 1))
        bias_2 = np.zeros((nb_outputs, 1))
        self.weights = []
        self.weights.append(weight_1)
        self.weights.append(weight_2)
        self.weights.append(bias_1)
        self.weights.append(bias_2)

    # Create all activation/gradient fontion we need
    def __softmax(self, z):
        return (np.exp(z) / np.sum(np.exp(z) + 1e-6, axis=0, keepdims=True))

    def __relu(sefl, z):
        return np.maximum(z, 0)

    def __relu_gradient(self, d_output, z):
        dz = np.array(d_output, copy=True)
        dz[z <= 0] = 0
        return dz

    # def __sigmoid(self, z):
        # return 1.0 / (1.0 + np.exp(-z))

    # def __sigmoid_gradient(self, X, dA):
        # return dA * X * (1.0 - X)

    def __forward_propagate(self, X, weight, bias, func):
        z = np.dot(weight, X) + bias
        if func == 'softmax':
            A = self.__softmax(z)
        elif func == 'relu':
            A = self.__relu(z)
        save = (X, weight, bias, z)
        return A, save

    def __back_propagate(self, d_output, output, y, save, func):
        A, weight, _, z = save
        m = A.shape[1]
        if func == 'softmax':
            error = output - y
        elif func == 'relu':
            error = self.__relu_gradient(d_output, z)
        d_weight = (1 / m) * np.dot(error, A.T)
        d_bias = (1 / m) * np.sum(error, axis=1, keepdims=True)
        dA = np.dot(weight.T, error)
        return dA, d_weight, d_bias

    def __gradient_descent(self, d_weight_1, d_bias_1, d_weight_2, d_bias_2, learning_rate):
        self.weights[0] -= learning_rate * d_weight_1
        self.weights[2] -= learning_rate * d_bias_1
        self.weights[1] -= learning_rate * d_weight_2
        self.weights[3] -= learning_rate * d_bias_2

    def __cost(self, y, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(y + 1e-6) + (1 - Y) * np.log(1 - y + 1e-6), keepdims=True, axis=1)
        cost = np.squeeze(cost)
        return cost

    def fit(self, X, y):
        epochs = 1000
        learning_rate = 0.01
        cost_save = []
        X, X_test, y, y_test = split_train_test(X, y)
        for i in range(epochs):
            # activate the network
            temp, save1 = self.__forward_propagate(X, self.weights[0], self.weights[2], 'relu')
            output, save2 = self.__forward_propagate(temp, self.weights[1], self.weights[3], 'softmax')
            # activate the network on test data
            temp, _ = self.__forward_propagate(X_test, self.weights[0], self.weights[2], 'relu')
            output_test, _ = self.__forward_propagate(temp, self.weights[1], self.weights[3], 'softmax')
            # compute cost
            loss = self.__cost(output, y)
            val_loss = self.__cost(output_test, y_test)
            # Compute backward propagation
            d_output = -(np.divide(y, output + 1e-6) - np.divide(1 - y, 1 - output + 1e-6))
            d_output, d_weight_2, d_bias_2 = self.__back_propagate(d_output, output, y, save2, 'softmax')
            _, d_weight_1, d_bias_1 = self.__back_propagate(d_output, output, y, save1, 'relu')
            # perform gradient descent to update the weights
            self.__gradient_descent(d_weight_1, d_bias_1, d_weight_2, d_bias_2, learning_rate)
            # Report the training error
            cost_save.append(loss)
            if (i % 50) == 0:
                print('loss: {} - val_loss: {}'.format(loss, val_loss))
        error = pd.Series(cost_save)
        fig.add_trace(go.Scatter(y=error, x=error.index))
        fig.show()
        print('Training complete')

    def predict(self, X):
        temp, _ = self.__forward_propagate(X.T, self.weights[0], self.weights[2], 'relu')
        y, _ = self.__forward_propagate(temp, self.weights[1], self.weights[3], 'softmax')
        return y.T

    def score(self, target, output, score_func='rmse'):
        if score_func == 'accuracy':
            output = np.where(output > 0.5, 1, 0)
            res = (output == target).mean()
        if score_func == 'entropy':
            log_likelihood = (-np.log(output.T[range(target.T.shape[0]), target.T]))
            res = (np.sum(log_likelihood) / target.T.shape[0] / 10000 )
        if score_func == 'rmse':
            res = np.average((target - output) ** 2)
        return res


#%%
from utils import load_data

if __name__ == "__main__":

    X, y = load_data()

    mlp = MLP()

    mlp.fit(X, y)

    output = mlp.predict(X)

    rmse = mlp.score(y, output)
    accuracy = mlp.score(y, output, score_func='accuracy')
    loss_entropy = mlp.score(y, output, score_func='entropy')
    print('Rmse: ', rmse)
    print('Accuracy: ', accuracy)
    print('Loss Entropy: ', loss_entropy)

# %%