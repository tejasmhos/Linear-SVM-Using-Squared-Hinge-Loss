"""
This module performs a demonstration of my linear svm with squared hinge
loss on a simulated dataset.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import linearsvm as svm

# creating our synthetic data

print("Generating and standardizing synthetic data\n")
X = np.zeros((300, 60))
X[0:150, :] = np.random.normal(scale=2, size=(150, 60))
X[150:300, :] = np.random.normal(scale=5, size=(150, 60))
y = np.zeros((300, 1))
y = np.asarray([1]*150 + [-1]*150)

# splitting our data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# standardizing our data
scaler = sklearn.preprocessing.StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

lambda_val = 1

print("Performing training for lambda value = 1\n")
betas, objectives = svm.mylinearsvm(lambda_val, 0.1, 200, X_train, y_train)

plt.figure(figsize=(12, 10))
plt.title("Objective function vs iteration")
plt.ylabel('Objective value', fontsize=14)
plt.xlabel('Iteration', fontsize=14)
plt.plot(objectives)
plt.show()

misclass_train = svm.calc_misclass(betas[-1], X_train, y_train)
misclass_test = svm.calc_misclass(betas[-1], X_train, y_train)

print("The misclassification error for the training and test set is",
      misclass_train, "&", misclass_test, "respectfully")

misclass_train_all, misclass_test_all = svm.calc_misclass_all_betas(
    betas, X_train, y_train, X_test, y_test)

svm.plot_misclass(misclass_train_all, misclass_test_all)





