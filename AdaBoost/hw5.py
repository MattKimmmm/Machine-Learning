#!/usr/bin/python3
# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use

    alphas = np.zeros(n_trees)
    g_train = np.zeros((n_trees, X_train.shape[0]))
    g_test = np.zeros((n_trees, X_test.shape[0]))

    dataSize = len(X_train)
    
    # initialize weights to 1/N
    weights = np.ones(dataSize) / dataSize

    for i in range(n_trees):
        # initialize a decision stump
        stump = DecisionTreeClassifier(criterion = 'entropy', max_depth=1)

        # calculate error
        error, tree = calc_error(X_train, y_train, stump, weights)
        g = tree.predict(X_train)
        g_testing = tree.predict(X_test)

        # calculate alpha
        alpha = 0.5 * np.log((1 - error) / error)

        # update weights
        weights_abnormal = np.multiply(weights, np.exp(-alpha * y_train * g))
        weights = normalize_weights(weights_abnormal)

        # save alpha and g
        alphas[i] = alpha
        g_train[i] = g
        g_test[i] = g_testing

    # calculate training error and testing error
    hypothesis_train = np.sign(np.dot(alphas, g_train))
    hypothesis_test = np.sign(np.dot(alphas, g_test))

    train_error = np.sum(hypothesis_train != y_train) / len(y_train)
    # print("training error: ", train_error)
    test_error = np.sum(hypothesis_test != y_test) / len(y_test)
    # print("testing error: ", test_error)

    return train_error, test_error
    # return train_error, test_error

# normalize weights
def normalize_weights(weights):
    return weights / np.sum(weights)

# calculate in-sample error
def calc_error(X, y, stump, weights):
    
    tree = stump.fit(X, y, sample_weight=weights)
    g = tree.predict(X)

    error = np.sum(np.multiply(weights, g != y))
    # print("in-sample error: ", error)

    return error, tree

    

def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_trees = 200

    # Split data
    train_1_3 = og_train_data[np.logical_or(og_train_data[:,0] == 1, og_train_data[:,0] == 3)]
    test_1_3 = og_test_data[np.logical_or(og_test_data[:,0] == 1, og_test_data[:,0] == 3)]
    
    X_train_1_3 = train_1_3[:,1:]
    y_train_1_3 = train_1_3[:,0]
    y_train_1_3 = np.where(y_train_1_3 == 1, 1, -1)

    X_test_1_3 = test_1_3[:,1:]
    y_test_1_3 = test_1_3[:,0]
    y_test_1_3 = np.where(y_test_1_3 == 1, 1, -1)

    train_3_5 = og_train_data[np.logical_or(og_train_data[:,0] == 3, og_train_data[:,0] == 5)]
    test_3_5 = og_test_data[np.logical_or(og_test_data[:,0] == 3, og_test_data[:,0] == 5)]

    X_train_3_5 = train_3_5[:,1:]
    y_train_3_5 = train_3_5[:,0]
    y_train_3_5 = np.where(y_train_3_5 == 3, 1, -1)

    X_test_3_5 = test_3_5[:,1:]
    y_test_3_5 = test_3_5[:,0]
    y_test_3_5 = np.where(y_test_3_5 == 3, 1, -1)

    # work on plots!! and fix the training error on dataset 1-3
    train_e_13_list = np.zeros(num_trees)
    test_e_13_list = np.zeros(num_trees)
    train_e_35_list = np.zeros(num_trees)
    test_e_35_list = np.zeros(num_trees)

    print("Starting AdaBoost...")
    for i in range(1, num_trees + 1):
        train_error_13, test_error_13 = adaboost_trees(X_train_1_3, y_train_1_3, X_test_1_3, y_test_1_3, i)
        print("1 vs 3: ", i, " trees, training error: ", train_error_13, " testing error: ", test_error_13)
        train_error_35, test_error_35 = adaboost_trees(X_train_3_5, y_train_3_5, X_test_3_5, y_test_3_5, i)
        print("3 vs 5: ", i, " trees, training error: ", train_error_35, " testing error: ", test_error_35)

        train_e_13_list[i - 1] = train_error_13
        test_e_13_list[i - 1] = test_error_13
        train_e_35_list[i - 1] = train_error_35
        test_e_35_list[i - 1] = test_error_35

    # Plot oob error vs number of bags
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(train_e_13_list, label='Train')
    axs[0].plot(test_e_13_list, label='Test')
    axs[0].legend(loc="upper right")
    axs[0].set_title('1 vs 3: Training Error vs Number of Trees')
    axs[0].set_ylabel('Error')
    axs[0].set_xlabel('Number of Trees')

    axs[1].plot(train_e_35_list, label='Train')
    axs[1].plot(test_e_35_list, label='Test')
    axs[1].legend(loc="upper right")
    axs[1].set_title('3 vs 5: Training Error vs Number of Trees')
    axs[1].set_ylabel('Error')
    axs[1].set_xlabel('Number of Trees')

    fig.tight_layout(pad=1.0)
    plt.show()

if __name__ == "__main__":
    main_hw5()
