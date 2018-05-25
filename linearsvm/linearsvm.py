"""
This module contains all the methods required to visualize a squared hinge
loss support vector machine.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
import scipy as sp
import statistics


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
    return -2/len(y)*(
       np.maximum(0,1-((y[:, np.newaxis]*X).dot(beta)))).dot(
       y[:, np.newaxis]*X) + 2 * lambdat * beta


def objective(beta, lambdat, X, y):
    return 1/len(y) * (
        np.sum((np.maximum(0, 1-((y[:, np.newaxis]*X).dot(beta)))**2)))\
           + lambdat * np.linalg.norm(beta)**2


def backtracking(beta, lambdat, t, X, y, alpha=0.5, frac=0.5, maxiter=100):
    grad_beta = compute_grad(beta, lambdat, X=X, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_t = False
    iter = 0
    while (not found_t) and iter < maxiter:
        if objective(beta - t * grad_beta, lambdat=lambdat, X=X, y=y)\
                < objective(beta, lambdat=lambdat, X=X, y=y)\
                - alpha * t * norm_grad_beta ** 2:
            found_t = True
        else:
            t *= frac
            iter += 1
    return t

