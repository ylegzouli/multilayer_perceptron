#%%
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

NB_INPUT = 3
NB_HIDDEN = [3, 3]
NB_OUTPUT = 2
EPOCH = 50
LR = 0.1

#%%

class MLP(BaseEstimator, TransformerMixin):
    
    def __init__(self, nb_inputs=NB_INPUT, nb_hidden=NB_HIDDEN, nb_outputs=NB_OUTPUT, 
                epochs=EPOCH, learning_rate=LR):
        # set NN parameters
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.epochs = epochs
        self.learning_rate=learning_rate
        # create list of layers
        layers = [self.nb_inputs] + self.nb_hidden + [self.nb_outputs]
        # init tab of random weights
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights
        # init tab of activations
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        # init tab of gradients
        grad = []
        for i in range(len(layers) - 1):
            g = np.zeros((layers[i], layers[i+1]))
            grad.append(g)
        self.grad = grad

    # create preactivation fonction (multipication between weight and input)
    def __pre_activation(self, input, weights):
        return np.dot(input, weights)

    # Create all activation/gradient fontion we need
    def __sigmoid_activation(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def __sigmoid_gradient(self, X):
        return X * (1.0 - X)

    def __forward_propagate(self, X):
        y = X
        self.activations[0] = y
        for i, w in enumerate(self.weights):
            # print (y.shape, w.shape)
            # calculate net input
            z = self.__pre_activation(y, w)
            # calculate activation
            y = self.__sigmoid_activation(z)
            self.activations[i + 1] = y
        return y

    def __back_propagate(self, error):
        # iterate backwards through the network layers
        for i in reversed(range(len(self.grad))):
            # get activation for previous layer
            activations = self.activations[i + 1]
            # apply sigmoid derivative function
            delta = error * self.__sigmoid_gradient(activations)
            # reshape delta to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T
            # get activations for current layer
            current_activations = self.activations[i]
            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)
            # save derivative after applying matrix multiplication
            self.grad[i] = np.dot(current_activations, delta_re)
            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

    def __gradient_descent(self, learningRate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.grad[i]
            weights += derivatives * learningRate
    
    # Score fonction
    def __mse(self, target, output):
        return np.average((target - output) ** 2)

    def fit(self, X, y):
        for i in range(self.epochs):
            sum_errors = 0
            # iterate through all the training data
            for j, x in enumerate(X):
                target = y[j]
                # activate the network
                output = self.__forward_propagate(x)
                error = target - output
                self.__back_propagate(error)
                # perform gradient descent to update the weights
                self.__gradient_descent(self.learning_rate)
                # keep track of the MSE for reporting later
                sum_errors += self.__mse(target, output)
            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(X), i+1))
        print('Training complete')

    def predict(self, X):
        y = self.__forward_propagate(X)
        return y

#%%
from random import random

if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])
    print(items.shape)
    print(targets.shape)

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2,[5], 1)

    # train network
    mlp.fit(items, targets)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.predict(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))

# %%
