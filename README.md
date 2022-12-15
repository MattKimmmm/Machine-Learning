# Machine-Learning

This repository includes Machine Learning Projects done in CSE 417, Washington University in St. Louis

1. Perceptron Learning Algorithm (PLA)
- Create a 11-dimensional, uniform, and randomized weight vectors and data set.
- Apply PLA on the dataset using randomly created vectors.
- Repeat the above for 1000 times.
- Analyze the number of iterations for each run, to be compared with calculated theoretical bounds.

2. Logistic Regression and Binary Classification Models
- Given data: Heart disease cases from Cleveland
- Initialize learning rate and maximum number of iterations.
- Utilize gradient descent to learn logistic regression model and binary classification model.
- Compare and analyze each run in terms of different learning rate and maximum number of iterations

3. Regularizations
- Given data: digits (classification between {1,6,9} and {0,7,8})
- Initialize learning rates and maximum number of iterations.
- Implement truncated gradient with L1 regularizer and a regular gradient descent with L2 regularizer for learning logistic regression model and binary classificatoin model.
- Compare and analyze the results in terms of different regularizers, kinds of gradient descent, and different learning rates.

4. Random Forrest
- Given data: handwritten digits (to be classified as: 1 vs 3, 3 vs 5)
- Initialize the number of bags to be used.
- Bootstrap samples from the dataset.
- Learn decision trees for each bag, and compute the out-of-bag errors.
- Compare and analyze each run in terms of different number of bags.

5. Adaptive Boosting (AdaBoost)
- Given data: handwritten digits (to be classified as: 1 vs 3, 3 vs 5)
- Initialize normalizing weight and the number of weak hypotheses (decision stump)
- Apply AdaBoost
- Analyze the trends of training / testing error with different number of hypotheses
