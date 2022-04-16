#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/16/20 9:10 PM 2020

@author: Anirban Das
"""
import numpy as np

EPSILON = 0.0000001


def sigmoid(z):
    m = np.exp(-z)
    return 1 / (1 + m)


def logistic_cost(theta, x, y, lambduh=0, penalise_intercept=False):
    h = sigmoid(x @ theta)
    m = len(y)
    # import pdb; pdb.set_trace()
    if penalise_intercept:
        cost_ = 1 / m * (-y.T @ np.log(h + EPSILON) - (1 - y).T @ np.log(1 - h + EPSILON)) + lambduh/2 *theta.T @ theta
    else:
        cost_ = 1 / m * (-y.T @ np.log(h + EPSILON) - (1 - y).T @ np.log(1 - h + EPSILON)) + lambduh/2 *theta[1:].T @ theta[1:]
    # grad = 1 / m * ((y - h) @ x)
    return cost_  # , grad


def squared_hinge_loss(theta, X, y, C=1, penalise_intercept=False):
    margin = np.maximum(0, 1 - np.multiply(y, X @ theta))
    """
    By default do not penalize intercept
    Deep Learning using Linear Support Vectors for ex. supports adding column of ones
    ISLR Equation 9.25 as well as the general form of SVM suggests only penalize
    parameters which are not the intercept.    
    """
    if penalise_intercept:
        cost_ = 0.5 * theta.T @ theta + C/2 * np.mean(margin) # in some cases use np.mean?
    else:
        cost_ = 0.5 * theta[1:].T @ theta[1:] + C/2 * np.mean(margin) # in some cases use np.mean?
    cost_ = np.sum(cost_)
    
    # g = theta - C/m * (X' * (margin .* Y));
    return cost_    

def linear_cost(theta, x, y, lambduh=0, penalise_intercept=False):
    """
    By default do not penalize intercept
    """
    m = len(x)
    residual = x@theta -y
    # this is ridge regression
    if penalise_intercept:
        cost_ = 1/(2*m) * (residual.T @ residual) + lambduh/2 * theta.T @ theta
    else:
        cost_ = 1/(2*m) * (residual.T @ residual) + lambduh/2 * theta[1:].T @ theta[1:]    
    return cost_[0,0]