import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys


# cost = (1/2m)*sum((h(thetaj)-yj)^2)
def cost(X, y, theta):
    m = X.shape[0]
    #print(y.shape)

    predictions = X.dot(theta)

    errors = predictions[0] - y

    sqErrors = np.power(errors, (2))

    cost = (1/(2*m))*np.sum(sqErrors)

    return cost


### Formula is...
### thetaj = thetaj - alpha*(1/m)*(X*theta-y)'*X
def gradientDescent(X, y, theta, alpha, iterations):
    """ """
    m = X.shape[0]
    J_History = []

    for _ in range(0, iterations):
        derivative = (1/m) * ((X @ theta).T - y) @ X
        theta = (theta.T - (alpha * derivative)).T
        J_History.append(cost(X, y, theta))

    return (theta, J_History)

def normalizeData(data):
    """ normalizes the data """

    mu = data.mean(axis=0)
    sigma = data.std(axis=0)
    data_norm = np.divide(data-mu, sigma)

    return (mu, sigma, data_norm)

def predict(X, theta, mu=None, sigma=None):
    """ predict values for some number of feature values

        expects that the features are *not* prepended with 1
    """
    # if supplied with mu/sigma - normalize
    if mu is not None and mu.any():
        X = X - mu

    if sigma is not None and sigma.any():
        X = np.divide(X, sigma)

    X = np.insert(X, 0, 1, axis=1)
    predictions = X @ theta

    return predictions

def normalEquation(X, y):
    """ """
    return np.linalg.inv(X.T @ X) @ X.T @ y

def learn(X, y, algType='gradientDescent', iterations=400):
    # arbitrary alpha, update by plotting J(theta)
    alpha = 0.13
    #set up our thetas = 0 for each feature
    theta = np.zeros(X.shape[1])

    if algType == 'gradientDescent':
        (theta, JHistory) = gradientDescent(X, y, theta, alpha, 250)
    elif algType == 'normalEquation':
        (theta, JHistory) = (normalEquation(X,y), None)

    return (theta, JHistory)
    
def main(path):
    # get data
    data = np.genfromtxt(path, delimiter=',')

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    numRows = data.shape[0]
    numCols = data.shape[1]

    # features are first n-1 cols
    X = data[:,0:numCols-1]
    (mu, sigma, X) = normalizeData(X)

    # prepend a row of 1s for our first feature
    C = np.ones((numRows, 1))
    X = np.append(C, X, axis=1)

    # y is last row
    y = data[:,-1]

    # get theta and history of costs - history lets us  
    (theta, J_History) = learn(X, y, algType='gradientDescent', iterations=1500)
    (theta2, J_History) = learn(X, y, algType='normalEquation')

    print("Theta from gradient descent:")
    print(theta)

    print("Theta from normal equation:")
    print(theta2)

    vals = np.array([[2000, 2], [1000, 3]])

    print("Predicting for feature(s)")
    print("{0}\n".format(vals))

    predictions = predict(vals, theta, mu, sigma)
    predictions2 = predict(vals, theta2, mu, sigma)

    vals1 = np.insert(vals, len(vals), predictions, axis=1)
    vals2 = np.insert(vals, len(vals), predictions2, axis=1)

    print("Prediction, gradient descent:")
    print(vals1)

    print("Prediction, normal equation:")
    print(vals2)

    exit()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage is {0} <datafile>".format(sys.argv[0]))
        exit()

    main(sys.argv[1])


