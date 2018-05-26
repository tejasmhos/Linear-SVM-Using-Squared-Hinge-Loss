"""
This module performs a comparision between my algorithm and Sklearn's
implementation. In order to facilitate this comparison, I use the spam
data set from The Elements of Statistical Learning. I compare the objective
value and the misclassification errors for the training and the testing data.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
import linearsvm as svm
from sklearn.svm import LinearSVC

# Downloading the data from the Stanford website
print("Downloading the data\n")

spam = pd.read_table('https://web.stanford.edu/~hastie/ElemStatLearn/'
                     'datasets/spam.data', sep=' ', header=None)
traintestind = pd.read_table('https://web.stanford.edu/~hastie/'
                             'ElemStatLearn/datasets/spam.traintest', sep=' ',
                               header=None)
X = np.asarray(spam)[:, 0:-1]

y = spam[57]
y = y.map({0: -1, 1: 1})
y = np.asarray(y)

# Splitting the data according to the split given on the website
indvar = np.array(traintestind).T[0]
X_train = X[indvar == 0, :]
X_test = X[indvar == 1, :]
y_train = y[indvar == 0]
y_test = y[indvar == 1]

print("Standardizing the data\n")

# Using Standard Scaler to scale the data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# First train using sklearn
print("Training Sklearn model\n")
sklearn_model = LinearSVC(penalty='l2', loss='squared_hinge', C=0.1)
sklearn_model.fit(X_train, y_train)
predictions_sklearn_test = sklearn_model.predict(X_test)
predictions_sklearn_train = sklearn_model.predict(X_train)
sklearn_coefficients = sklearn_model.coef_
misclass_train_sklearn = 1 - np.mean(predictions_sklearn_train == y_train)
misclass_test_sklearn = 1 - np.mean(predictions_sklearn_test == y_test)


# Perform training on my own model
print("Training my own model\n")
betas, objs = svm.mylinearsvm(0.1, 0.1, 100, X_train, y_train)
misclass_train_own = svm.calc_misclass(betas[-1], X_train, y_train)
misclass_test_own = svm.calc_misclass(betas[-1], X_test, y_test)

objval_sk = svm.objective(sklearn_coefficients.T, 0.1, X_train, y_train)

# Output results to stdout
print("Outputting data\n")
print("The value of the objective for my implementation is:", objs[-1])
print("The value of the objective for sklearn implementation is:", objval_sk)
print("The misclassification error for training set using sklearn is:", misclass_train_sklearn)
print("The misclassification error for test set using sklearn is:", misclass_test_sklearn)
print("The misclassification error for the training set using my own ", misclass_train_own)
print("The misclassification error for the test set using my own", misclass_test_own)
