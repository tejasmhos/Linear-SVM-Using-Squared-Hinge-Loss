"""
This module performs a demonstration of my linear svm with squared hinge
loss on a real dataset. We use the spam dataset, from The
Elements of Statistical Learning

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

# load the data
print('Downloading the data\n')
spam = pd.read_table('https://web.stanford.edu/~hastie/ElemStatLearn/'
                     'datasets/spam.data', sep=' ', header=None)
traintestind = pd.read_table('https://web.stanford.edu/~hastie/'
                             'ElemStatLearn/datasets/spam.traintest', sep=' ',
                               header=None)
X = np.asarray(spam)[:, 0:-1]

y = spam[57]
y = y.map({0: -1, 1: 1})
y = np.asarray(y)

indvar = np.array(traintestind).T[0]
X_train = X[indvar == 0, :]
X_test = X[indvar == 1, :]
y_train = y[indvar == 0]
y_test = y[indvar == 1]

# standardizing the data
print('Scaling the data\n')
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print('Training the model for lambda = 1\n')
lambda_val = 10
betas, objective_vals = svm.mylinearsvm(lambda_val, 0.1, 100, X_train, y_train)

plt.figure(figsize=(12, 10))
plt.title("Objective function vs iteration")
plt.ylabel('Objective value', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.plot(objective_vals)
plt.show()

misclass_train = svm.calc_misclass(betas[-1], X_train, y_train)
misclass_test = svm.calc_misclass(betas[-1], X_train, y_train)

misclass_train_all, misclass_test_all = svm.calc_misclass_all_betas(
    betas, X_train, y_train, X_test, y_test)

print("The misclassification error for the training and test set is:",
      misclass_train, "&", misclass_test, "respectively\n")

svm.plot_misclass(misclass_train_all, misclass_test_all)

print("Performing cross validation to find the best lambda value\n")

lambda_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
best_lambda = svm.crossvalidation(X_train, y_train, 3, lambda_values)

print("Training the model using this lambda value\n")

betas_cv, objective_vals_cv = svm.mylinearsvm(lambda_val, 0.1, 100, X_train, y_train)

plt.figure(figsize=(12, 10))
plt.title("Objective function vs iteration")
plt.ylabel('Objective value', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.plot(objective_vals_cv)
plt.show()

misclass_train_cv = svm.calc_misclass(betas_cv[-1], X_train, y_train)
misclass_test_cv = svm.calc_misclass(betas_cv[-1], X_train, y_train)

misclass_train_all_cv, misclass_test_all_cv = svm.calc_misclass_all_betas(
    betas_cv, X_train, y_train, X_test, y_test)

print("The misclassification error for the training and test set is:",
      misclass_train_cv, "&", misclass_test_cv, "respectively\n")

svm.plot_misclass(misclass_train_all_cv, misclass_test_all_cv)

print("Thanks for demoing\n")




