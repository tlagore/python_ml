import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


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
        #derivative = (1/m)*(X.dot(theta)[0] - y).T.dot(X)
        derivative = (1/m) * ((X @ theta)[0] - y).T @ X
        theta = theta[0] - (alpha * derivative)
        J_History.append(cost(X, y, theta))

    return (theta, J_History)


def main():
    # get data
    path = os. getcwd() + '\\ex1data1.txt'
    data = np.genfromtxt(path, delimiter=',')

    numRows = data.shape[0]
    numCols = data.shape[1]

    # arbitrary alpha, will update later
    alpha = 0.003

    # features are first n-1 cols
    X = data[:,0:numCols-1]
    C = np.ones((numRows, 1))

    # prepend a row of 1s for our first feature
    X = np.append(C, X, axis=1)

    #set up our thetas = 0 for each feature
    theta = np.zeros((numCols, 1))

    # y is last row
    y = data[:,-1]

    # get theta and history of costs - history lets us  
    (theta, J_History) = gradientDescent(X, y, theta, alpha, 400)

    plt.plot(J_History)
    plt.show()

    print(theta)

if __name__ == "__main__":
    main()


