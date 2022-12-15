#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
# import matplotlib.pyplot as plt
# import graphviz
from scipy import stats


def single_decision_tree(train, test):

    X_train = train[:, 1:]
    y_train = train[:, 0]
    X_test = test[:, 1:]
    y_test = test[:, 0]

    # Learn a single decision tree
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    clf = clf.fit(X_train, y_train)
    print(clf)
    
    # tree_plot = tree.plot_tree(clf)
    x_train = clf.predict(X_train)
    x = clf.predict(X_test)

    # Calculate errors
    binary_error_train = np.sum(x_train != y_train) / len(y_train)
    binary_error_test = np.sum(x != y_test) / len(y_test)

    return binary_error_train, binary_error_test

def bagged_trees(train, test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function

    # out_of_bag_error =
    # test_error =

    # create bootstrap samples
    trees = []
    indices = []

    clf = tree.DecisionTreeClassifier(criterion = 'entropy')

    # get indices for oob data points
    # print("Generating ", num_bags, "bags...")
    for i in range(num_bags):
        sample, indices_boot = bootstrap(train)
        trees.append(clf.fit(sample[:, 1:], sample[:, 0]))

        # train.shape = (1663, 257)
        # indices_boot.shape = (1663,)
        oob_indices = np.setdiff1d(np.arange(train.shape[0]), indices_boot)
        indices.append(oob_indices)

    # indices.shape = (10,)

    # calculate oob error
    predictions_aggregated = []
    for i in range(train.shape[0]):
        predictions = []
        for count, index in enumerate(indices):
            
            if i not in index:
                predictions.append(trees[count].predict(train[i, 1:].reshape(1, -1)))
        
        if len(predictions) != 0:
            g = stats.mode(predictions, keepdims=False)
            predictions_aggregated.append(g[0][0])
        else:
            predictions_aggregated.append(None)

    predictions_aggregated = np.array(predictions_aggregated)
    out_of_bag_error = np.sum(predictions_aggregated != train[:, 0]) / len(train[:, 0])
    # print("Out of bag error: ", out_of_bag_error)

    # calculate test error
    predictions_aggregated = []
    for i in range(test.shape[0]):
        predictions = []
        for count, index in enumerate(indices):
            predictions.append(trees[count].predict(test[i, 1:].reshape(1, -1)))
        
        g = stats.mode(predictions, keepdims=False)
        predictions_aggregated.append(g[0][0])

    predictions_aggregated = np.array(predictions_aggregated)
    test_error = np.sum(predictions_aggregated != test[:, 0]) / len(test[:, 0])


    return out_of_bag_error, test_error

def bootstrap(train):
    sample = np.zeros((train.shape[0], train.shape[1]))
    indices = []

    for i in range(train.shape[0]):
        index = np.random.choice(train.shape[0], replace=True)
        indices.append(index)
        sample[i] = train[index]

    # indices = np.array(indices)
    return sample, indices

    

def main_hw4():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    # num_bags =

    # Split data
    # Create training datasets
    train_1_3 = og_train_data[np.logical_or(og_train_data[:,0] == 1, og_train_data[:,0] == 3)]
    train_3_5 = og_train_data[np.logical_or(og_train_data[:,0] == 3, og_train_data[:,0] == 5)]

    # Create testing datasets

    test_1_3 = og_test_data[np.logical_or(og_test_data[:,0] == 1, og_test_data[:,0] == 3)]
    test_3_5 = og_test_data[np.logical_or(og_test_data[:,0] == 3, og_test_data[:,0] == 5)]

    # create 200 bags for each dataset
    bags_1_3 = []
    bags_3_5 = []
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')

    for i in range(200):
        # sample shape: (1663, 257)
        sample = train_1_3[np.random.choice(train_1_3.shape[0], train_1_3.shape[0], replace=True)]
        bags_1_3.append(sample)

    for i in range(200):
        # sample shape: (1663, 257)
        sample = train_3_5[np.random.choice(train_3_5.shape[0], train_3_5.shape[0], replace=True)]
        bags_3_5.append(sample)

    # Run bagged trees on 1 vs 3 for 1 ~ 200 bags
    oob_13 = []
    oob_35 = []
    for i in range(199):
        out_of_bag_error, test_error = bagged_trees(train_1_3, test_1_3, i+1)
        oob_13.append(out_of_bag_error)
        print("1 vs 3: ", i+1, " bags, oob error: ", out_of_bag_error, " test error: ", test_error)

    # # # Run bagged trees on 3 vs 5 for 1 ~ 200 bags
    for i in range(199):
        out_of_bag_error, test_error = bagged_trees(train_1_3, test_1_3, i+1)
        oob_35.append(out_of_bag_error)
        print("3 vs 5: ", i+1, " bags, oob error: ", out_of_bag_error, " test error: ", test_error)

    print("starting...")

    out_of_bag_error, test_error = bagged_trees(train_1_3, test_1_3, 200)
    print("1 vs 3: 200 bags, oob error: ", out_of_bag_error, " test error: ", test_error)
    out_of_bag_error, test_error = bagged_trees(train_3_5, test_3_5, 200)
    print("3 vs 5: 200 bags, oob error: ", out_of_bag_error, " test error: ", test_error)

    # # Run single decision tree on 1 vs 3 and 3 vs 5
    train_error_13, test_error_13 = single_decision_tree(train_1_3, test_1_3)
    train_error_35, test_error_35 = single_decision_tree(train_3_5, test_3_5)
    print("1 vs 3: train error: ", train_error_13, " test error: ", test_error_13)
    print("3 vs 5: train error: ", train_error_35, " test error: ", test_error_35)

    # Plot oob error vs number of bags
    fig, axs = plt.subplots(2)

    axs[0].plot(oob_13)
    axs[0].set_title('1 vs 3')
    axs[0].set_ylabel('Out of Bag Error')
    axs[0].set_xlabel('Number of Bags')

    axs[1].plot(oob_35)
    axs[1].set_title('3 vs 5')
    axs[1].set_ylabel('Out of Bag Error')
    axs[1].set_xlabel('Number of Bags')

    fig.tight_layout(pad=1.0)
    plt.show()


if __name__ == "__main__":
    main_hw4()

