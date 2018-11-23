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

    return data_norm


def main(path):
    # get data
    data = np.genfromtxt(path, delimiter=',')

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    numRows = data.shape[0]
    numCols = data.shape[1]

    # arbitrary alpha, will update later
    alpha = 0.01

    # features are first n-1 cols
    X = data[:,0:numCols-1]
    X = normalizeData(X)

    # prepend a row of 1s for our first feature
    C = np.ones((numRows, 1))
    X = np.append(C, X, axis=1)

    #set up our thetas = 0 for each feature
    theta = np.zeros((numCols, 1))

    # y is last row
    y = data[:,-1]

    # get theta and history of costs - history lets us  
    (theta, J_History) = gradientDescent(X, y, theta, alpha, 5)
    print(theta)

    exit()
    plt.plot(J_History)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage is {0} <datafile>".format(sys.argv[0]))
        exit()

    main(sys.argv[1])


