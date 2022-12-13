#!/usr/bin/python3
# Homework 2 Code
import numpy as np
import pandas as pd
import time


def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:

    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # calculate the binary classification error of w on the data set (X, y)
    binary_error = np.sum(np.sign(sigmoid(np.dot(X, w)) -.5) != y) / len(y)


    return binary_error



def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:
    
    w = w_init
    t = 0
    
    # add 1s to the beginning of X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    while t < max_its:
        t += 1
        g = gradient(X, y, w)
        w = w - eta * g

        if np.max(np.absolute(g)) < grad_threshold:
            break
    
    # calculate cross-entropy error
    theta = sigmoid(np.multiply(y, np.dot(X, w)))
    e_in = np.sum(np.log(1/theta)) / len(y)

    return t, w, e_in

# define sigmoid function
def sigmoid(score):
    return 1 / (1 + np.exp(-score))

# calculate the gradient
def gradient(X, y, w):

    k = np.multiply(np.reshape(y, (-1,1)), X)
    gTotal = np.sum(np.multiply(-k, sigmoid(np.dot(-k, np.reshape(w, (-1,1))))), axis=0)


    return gTotal/len(y)

# normalize the x data
def scale(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_test, axis=0)
    
    X_train = (X_train - mean) /std
    X_test = (X_test - mean) / std

    return X_train, X_test

def main():
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')
    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')

    # Replace 0 with -1 for heartdisease::category|0|1 in both train_data and test_data
    train_data['heartdisease::category|0|1'].replace(0, -1, inplace=True)
    test_data['heartdisease::category|0|1'].replace(0, -1, inplace=True)

    # Extract heartdisease::category|0|1 as y_train and y_test and drop them from train_data and test_data
    y_train = train_data['heartdisease::category|0|1']
    y_test = test_data['heartdisease::category|0|1']

    train_data.drop('heartdisease::category|0|1', axis=1, inplace=True)
    test_data.drop('heartdisease::category|0|1', axis=1, inplace=True)

    X_test = np.array(test_data)
    X_train = np.array(train_data)
    y_test = np.array(y_test)
    y_train = np.array(y_train)

    # initialize w and other variables
    w_init = np.zeros(train_data.shape[1] + 1)
    eta = 10**-5
    grad_threshold = 10**-3
    max_its = 10**6

    # call Part A function
    partA(X_train, y_train, w_init, eta, grad_threshold, X_test, y_test)

    # call Part B function
    partB(X_train, y_train, w_init, max_its, grad_threshold, X_test, y_test)
    

def partA(X_train, y_train, w_init, eta, grad_threshold, X_test, y_test):

    for its in range(4,7):
        tic = time.perf_counter()
        max_its = 10**(its)
        t, w, e_in = logistic_reg(X_train, y_train, w_init, max_its, eta, grad_threshold)
        print ('iterations = ', t)
        print ('weight vector = ', w)
        print ('Cross-entropy error = ', e_in)
        print ('Binary classification error on training data= ', find_binary_error(w, X_train, y_train))
        print ('Binary classification error on test data= ', find_binary_error(w, X_test, y_test))

        toc = time.perf_counter()
        print(f"Time taken for max_its = 10^{its} : {toc - tic:0.4f} seconds")

def partB(X_train, y_train, w_init, max_its, grad_threshold, X_test, y_test):

    X_train ,X_test = scale(X_train, X_test)
    eta_list = [0.01, 0.1, 1, 4, 7, 7.5, 7.6, 7.7]
    grad_threshold = 10**-6

    for eta in eta_list:
        
        tic = time.perf_counter()
        t, w, e_in = logistic_reg(X_train, y_train, w_init, max_its, eta, grad_threshold)
        print ('For eta = ', eta)
        print ('iterations = ', t)
        print ('weight vector = ', w)
        print ('Cross-entropy error = ', e_in)
        print ('Binary classification error = ', find_binary_error(w, X_test, y_test))

        toc = time.perf_counter()
        print(f"Time taken for eta = 10^{eta} : {toc - tic:0.4f} seconds")



if __name__ == "__main__":
    main()
