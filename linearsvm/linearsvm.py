"""
This module contains all the methods required to visualize a squared hinge
loss support vector machine. There are a set of functions that are used
by the demo files to perform the machine learning task.

This code was created by Tejas Hosangadi, and is licensed under the MIT license.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
import scipy as sp
import statistics
from sklearn.metrics import mean_squared_error


def compute_grad(beta, lambdat, X, y):
    """
    This function computes the gradient of the squared hinge loss
    objective function

    :param beta: A dx1 vector of beta values
    :param lambdat: Value of the regularization parameter
    :param X: A dxn matrix of features
    :param y: A nx1 vector of labels
    :return: A dx1 vector containing the gradient
    """
    return -2/len(y)*(np.maximum(0, 1-(
        (y[:, np.newaxis]*X).dot(beta)))).dot(
        y[:, np.newaxis]*X) + 2 * lambdat * beta


def objective(beta, lambdat, X, y):
    """
    This function computes the value of the objective function for the squared
    hinge loss. It is vectorized so as to enable faster computation.
    :param beta: A dx1 vector of beta values
    :param lambdat: Value of the regularization parameter
    :param X: A dxn matrix of features
    :param y: A nx1 vector of labels
    :return: A float value, equivalent to the value of the objective function
    """
    return 1/len(y) * (np.sum(
        (np.maximum(0, 1-((y[:, np.newaxis]*X).dot(beta)))**2)))\
           + lambdat * np.linalg.norm(beta)**2


def backtracking(beta, lambdat, t, X, y, alpha=0.5, frac=0.5, maxiter=100):
    """
    This function performs the backtracking operation to find the best step
    size.
    :param beta: A dx1 vector of beta values
    :param lambdat: Value of the regularization parameter
    :param t: initial step size
    :param X: A dxn matrix of features
    :param y: A nx1 vector of labels
    :param alpha: Constant used to define sufficient decrease condition
    :param frac: Fraction by which we decrease t if the previous t doesn't work
    :param maxiter: Maximum number of iterations to run our algorithm
    :return: Step size to use
    """
    grad_beta = compute_grad(beta, lambdat, X=X, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_t = False
    iter = 0
    while (not found_t) and iter < maxiter:
        if objective(beta - t * grad_beta, lambdat=lambdat, X=X, y=y) < objective(beta, lambdat=lambdat, X=X, y=y) - alpha * t * norm_grad_beta ** 2:
            found_t = True
        else:
            t *= frac
            iter += 1
    return t


def fast_grad(beta_init, theta_init, lambdat, t_init, maxiter, X, y):
    """
    This function implements the fast gradient algorithm, as developed by
    Yurii Nesterov.
    :param beta_init: The initial set of betas.
    :param theta_init: The initial set of thetas.
    :param lambdat: The regularization parameter
    :param t_init: The initial step size
    :param maxiter: Maximum iterations to run our algorithm
    :param X: A dxn matrix of features
    :param y: A nx1 vector of labels
    :return: The final beta values, and the objective values.
    """
    beta = beta_init
    theta = theta_init
    grad = compute_grad(theta, lambdat=lambdat, X=X, y=y)
    beta_vals = beta
    theta_vals = theta
    iter = 0
    obj_vals = []
    while iter < maxiter:
        t = backtracking(theta, lambdat=lambdat, t=t_init, X=X, y=y)
        beta_new = theta - t*grad
        theta = beta_new + iter/(iter+3)*(beta_new-beta)
        beta_vals = np.vstack((beta_vals, beta_new))
        obj_vals.append(objective(beta_new,lambdat,X=X,y=y))
        grad = compute_grad(theta, lambdat=lambdat, X=X, y=y)
        beta = beta_new
        iter += 1
    return beta_vals, obj_vals


def mylinearsvm(lambdat, eta_init, maxiter, X, y):
    """
    This function serves as a wrapper around the fast_grad function that we implemented above.

    :param lambdat: This is the regularization parameter.
    :param eta_init: This is the initial step size.
    :param maxiter: Maximum number of iterations to run the algorithm.
    :param X: A dxn matrix of features
    :param y: A nx1 vector of labels
    :return:
    """
    d = np.size(X, 1)
    beta_init = np.zeros(d)
    theta_init = np.zeros(d)
    betas, objs = fast_grad(beta_init, theta_init, lambdat, eta_init, maxiter,X=X,y=y)
    return betas, objs


def calc_misclass(beta_final, X, y):
    """
    This function calculations the misclassification error, given the final
    beta values, the X array, and the y array
    :param beta_final: The vector of final beta values
    :param X: A dxn matrix of features
    :param y: A nx1 vector of labels
    :return: Misclassification error
    """
    y_pred = np.dot(X, beta_final)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1
    num = np.count_nonzero(y-y_pred)
    err = num/float(X.shape[0])
    return err


def calc_misclass_all_betas(beta, train_feat, train_lab, test_feat, test_lab):
    """
    This function returns the misclassification error across all the betas
    for all the iterations that are run.
    :param beta: The array of betas for max_iter iterations
    :param train_feat: A dxn matrix of training features
    :param train_lab: An array of training labels
    :param test_feat: A dxn matrix of test features
    :param test_lab: An array of test labels
    :return: misclassfication errors for training and test sets
    """
    n, d = beta.shape
    mis_train = []
    mis_test = []
    for i in range(n):
        mis_train.append(calc_misclass(beta[i], train_feat, train_lab))
    for i in range(n):
        mis_test.append(calc_misclass(beta[i], test_feat, test_lab))
    return mis_train, mis_test


def plot_misclass(train, test):
    """
    This function is used to plot the misclassification error, uses
    matplotlib as the underlying plotting library.
    :param train: Train misclassification, array
    :param test: Test misclassification, array
    :return: Plot
    """
    plt.figure(figsize=(12, 10))
    ptrain,  = plt.plot(train, label='Train Misclassification')
    ptest, = plt.plot(test, label='Test Misclassification')
    plt.legend(handles=[ptrain, ptest], fontsize=14)
    plt.title('Training vs Test Misclassification', fontsize=16)
    plt.ylabel('Misclassification', fontsize=14)
    plt.xlabel('Iteration', fontsize=14)
    plt.show()


def crossvalidation(X, y, folds, lambdavals):
    """
    This function performs cross validation on the given data
    to find the best lambda value. The lambda values
    can be passed manually, as arguments.
    :param X: A dxn matrix of features
    :param y: A nx1 vector of labels
    :param folds: Number of folds
    :param lambdavals: Lambda values to use
    :return:
    """
    n = X.shape[0]
    Errors = np.empty((0, len(lambdavals)))
    index = (list(range(folds)) * (n//folds+1))[0:n]
    np.random.shuffle(index)
    index = np.array(index)
    for i in range(folds):
        X_train_CV = X[index != i, :]
        X_test_CV = X[index == i, :]
        y_train_CV = y[index != i]
        y_test_CV = y[index == i]
        Errorsinter = []
        for lam in lambdavals:
            betas, _ = mylinearsvm(lam, 0.1, 100, X_train_CV, y_train_CV)
            y_pred = np.dot(X_test_CV, betas[-1])
            Errorsinter.append(mean_squared_error(y_test_CV, y_pred))
        Errors = np.vstack((Errors, Errorsinter))
    mean_errors = np.mean(Errors, axis = 0)
    minimum_val = np.max(np.where(mean_errors == mean_errors.min()))
    lambda_best = lambdavals[minimum_val]
    print("The best value of lambda is:", lambda_best)
    return lambda_best




