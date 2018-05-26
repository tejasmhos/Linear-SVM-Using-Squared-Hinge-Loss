"""
This module demos my square hinged loss based SVM on some synthetic data.
The algorithm plots the objective values, the misclassification error
for the train and test sets, and also performs cross validation

Syntax:

    $> python real_data_demo.py

When the demo is run, the textual output is shown in the terminal
window, while the plots are generated in their own windows.
After a plot is viewed, closing the plot will generate the next
plot.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
import linearsvm as svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load the data
print("Synthesizing the data:\n")

X = np.zeros((200, 40))
X[0:100, :] = np.random.normal(scale=1, size=(100, 40))
X[100:200, :] = np.random.normal(loc=2, scale=10, size=(100, 40))
y = np.asarray([1]*100 + [-1]*100)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# standardizing the data
print('Scaling the data\n')
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print('Training the model for lambda = 10\n')
lambda_val = 10
betas, objective_vals = svm.mylinearsvm(lambda_val, 0.1, 100, X_train, y_train)

plt.figure(figsize=(12, 10))
plt.title("Objective function vs iteration")
plt.ylabel('Objective value', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.plot(objective_vals)
plt.show()

misclass_train = svm.calc_misclass(betas[-1], X_train, y_train)
misclass_test = svm.calc_misclass(betas[-1], X_test, y_test)

misclass_train_all, misclass_test_all = svm.calc_misclass_all_betas(
    betas, X_train, y_train, X_test, y_test)

print("The misclassification error for the training and test set is:",
      misclass_train_all[-1], "&", misclass_test_all[-1], "respectively\n")

svm.plot_misclass(misclass_train_all, misclass_test_all)

print("Performing cross validation to find the best lambda value\n")

lambda_values = [0.001, 0.01, 0.1, 1]
best_lambda = svm.crossvalidation(X_train, y_train, 3, lambda_values)

print("Training the model using this lambda value\n")

betas_cv, objective_vals_cv = svm.mylinearsvm(best_lambda, 0.1, 100, X_train, y_train)

plt.figure(figsize=(12, 10))
plt.title("Objective function vs iteration")
plt.ylabel('Objective value', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.plot(objective_vals_cv)
plt.show()

misclass_train_cv = svm.calc_misclass(betas_cv[-1], X_train, y_train)
misclass_test_cv = svm.calc_misclass(betas_cv[-1], X_test, y_test)

misclass_train_all_cv, misclass_test_all_cv = svm.calc_misclass_all_betas(
    betas_cv, X_train, y_train, X_test, y_test)

print("The misclassification error for the training and test set is:",
      misclass_train_all_cv[-1], "&", misclass_test_all_cv[-1], "respectively\n")

svm.plot_misclass(misclass_train_all_cv, misclass_test_all_cv)

print("Process complete. Exiting\n ")
