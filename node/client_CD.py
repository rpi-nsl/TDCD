#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:31:57 2020

@author: Anirban Das
"""

import numpy as np
import copy
from tqdm import tqdm
from node.utilities.cost import linear_cost, logistic_cost, sigmoid, squared_hinge_loss
import logging
logging.basicConfig(level=logging.DEBUG)


def cost(type_of_cost):
    if type_of_cost == "linear":
        return linear_cost
    if type_of_cost == "squared_hinge_loss":
        return squared_hinge_loss
    elif type_of_cost == "logistic":
        return logistic_cost


class ClientWorkerCoordinateDescent(object):
    """
    Base class for the clients which will do the training
    """

    def __init__(self, n_epochs: int, batch_size: int, learning_rate: float, X: np.array, y: np.array,
                 offset: float, device_index: int, dc_index: int, type_of_cost: str, lambduh: float,
                 start_feature_idx: int = 0, decreasing_step=False, penalize_intercept=False):
        """

        Args:
            n_epochs: Number of Epochs
            batch_size: The size of each batch
            learning_rate: The learning rate
            X: The feature set
            y: The labels
            offset: Offset= number of datapoints per client
            model: The DNN model : nn.Module object
            client_index: The index of the client in the datacenter
            dc_index: The index of the datacenter in the graph
        """
        self.max_iter = n_epochs
        self.batch_size = batch_size
        self.alpha = learning_rate
        self.offset = offset
        self.device_index = device_index
        self.dc_index = dc_index
        self.X = X  # [self.client_index * self.offset:(self.client_index + 1) * self.offset]
        self.y = y  # [self.client_index * self.offset:(self.client_index + 1) * self.offset]
        self.theta = np.zeros((self.X.shape[1], 1))
        self.costs = np.zeros(self.max_iter)
        self.lambduh = lambduh
        self.start_feature_idx = start_feature_idx
        self.cost = cost(type_of_cost)
        self.Xtheta = np.zeros((self.X.shape[0], 1))
        self.decreasing_step = decreasing_step
        self.penalize_intercept = penalize_intercept

    # @profile
    def train(self, global_step_no, local_epochs=0):
        logging.info(msg=f"Starting training for global level {global_step_no}")
        if local_epochs > 0:
            self.max_iter = local_epochs
        # Isolate the H_-k from other datacenters for the same label space
        # Obtained in the last iteration
        Xtheta_from_other_DC = self.Xtheta - self.X @ self.theta  # Assuming sample space is same
        for Q_idx, Q in enumerate(range(self.max_iter)):
            # batch gradient descent for the time being

            # If NO partital gradient information from outside is used
            # grad = 1/len(device.X) * device.X.T @ (device.X @ device.theta - device.y)

            # If partital gradient information from outside is used
            grad = 1 / len(self.X) * self.X.T @ (
                    (Xtheta_from_other_DC + self.X @ self.theta) - self.y) + self.lambduh * self.theta
            if self.decreasing_step:
                self.theta = self.theta - self.alpha / np.sqrt(global_step_no + 1) * grad  # decreasing step size
            else:
                self.theta = self.theta - self.alpha * grad

    def set_model(self, global_model):
        """
        Replace the local model with the global model and reset the optimizer with the parameters of the new model

        Args:
            global_model: The global model as nn.Module object

        Returns: None

        """
        self.theta = copy.deepcopy(global_model)
        self.costs = np.zeros(self.max_iter)

    def set_Xtheta(self, Xtheta):
        """
        Replace the local Xtheta with the global Xtheta

        Args:
            Xtheta: The global value of Xtheta for the corresponding data points in self.X

        Returns: None

        """
        self.Xtheta = copy.deepcopy(Xtheta)

    def get_model(self):
        return self.theta

    def get_xtheta(self):
        return self.Xtheta

    def calculate_and_get_Xtheta(self):
        self.Xtheta = self.X @ self.theta
        return self.Xtheta

    def train_error_ad_loss(self):
        pass


class LinearCDClient(ClientWorkerCoordinateDescent):
    """
    Class for the clients doing linear regression
    """

    def __init__(self, n_epochs: int, batch_size: int, learning_rate: float, X: np.array, y: np.array,
                 offset: float, device_index: int, dc_index: int, type_of_cost: str, lambduh: float,
                 start_feature_idx: int = 0, decreasing_step=False, penalize_intercept=False, **kwargs):
        super(LinearCDClient, self).__init__(n_epochs, batch_size, learning_rate, X, y,
                                             offset, device_index, dc_index, type_of_cost, lambduh,
                                             start_feature_idx, decreasing_step, penalize_intercept)

    def train(self, global_step_no, local_epochs=0):
        #logging.info(msg=f"Starting training for global level {global_step_no}")
        if local_epochs > 0:
            self.max_iter = local_epochs
        # Isolate the H_-k from other datacenters for the same label space
        # Obtained in the last iteration
        Xtheta_from_other_DC = self.Xtheta - self.X @ self.theta  # Assuming sample space is same
        for Q_idx, Q in enumerate(range(self.max_iter)):
            # batch gradient descent for the time being

            # If NO partital gradient information from outside is used
            # grad = 1/len(device.X) * device.X.T @ (device.X @ device.theta - device.y)

            # If partital gradient information from outside is used
            # By default do not penalize the intercept to prevent bias ISLR:
            if self.penalize_intercept:
                grad = 1 / len(self.X) * self.X.T @ (
                        (Xtheta_from_other_DC + self.X @ self.theta) - self.y) + self.lambduh * self.theta
            else:
                reg_term_without_intercept = self.lambduh * copy.deepcopy(self.theta) 
                reg_term_without_intercept[0] = 0
                grad = 1 / len(self.X) * self.X.T @ (
                        (Xtheta_from_other_DC + self.X @ self.theta) - self.y) + reg_term_without_intercept

            if self.decreasing_step:
                self.theta = self.theta - self.alpha / np.sqrt(global_step_no + 1) * grad  # decreasing step size
            else:
                self.theta = self.theta - self.alpha * grad


class LogisticCDClient(ClientWorkerCoordinateDescent):
    """
    Class for the clients doing Logistic regression
    """

    def __init__(self, n_epochs: int, batch_size: int, learning_rate: float, X: np.array, y: np.array,
                 offset: float, device_index: int, dc_index: int, type_of_cost: str, lambduh: float,
                 start_feature_idx: int = 0, decreasing_step=False, penalize_intercept=False, **kwargs):
        super(LogisticCDClient, self).__init__(n_epochs, batch_size, learning_rate, X, y,
                                               offset, device_index, dc_index, type_of_cost, lambduh,
                                               start_feature_idx, decreasing_step, penalize_intercept)

    def train(self, global_step_no, local_epochs=0):
        #logging.info(msg=f"Starting training for global level {global_step_no}")
        if local_epochs > 0:
            self.max_iter = local_epochs
        # TODO: FIX THIS
        Xtheta_from_other_DC = self.Xtheta - self.X @ self.theta  # Assuming sample space is same
        for iteration in range(self.max_iter):
            old_theta = copy.deepcopy(self.theta)
            h = sigmoid(Xtheta_from_other_DC + self.X @ self.theta)
            m = len(self.X)
            if self.penalize_intercept:
                grad = - 1 / m * self.X.T @ (self.y - h) + self.lambduh * self.theta
            else:
                reg_term_without_intercept = self.lambduh * copy.deepcopy(self.theta) 
                reg_term_without_intercept[0] = 0
                grad = - 1 / m * self.X.T @ (self.y - h) + reg_term_without_intercept

            self.theta = self.theta - self.alpha * grad

            # Idea from https://arxiv.org/pdf/1610.00040.pdf page 24
#            ex = ex * np.exp(-self.X @ (self.theta - old_theta))

        train_loss = self.cost(self.theta, self.X, self.y)
        return train_loss


class SVMCDClient(ClientWorkerCoordinateDescent):
    """
    Class for the clients participating in solving Support Vector Machine Primal form via Coordinate Descent
    We use L2 SVM or SVM with squared Hinge loss, with mean of hinge loss
    The objective function is as follows: 
      L(w) =  1/2||w||^2 + C/(2S) Σ_{j=1}^D [ max{0, 1- w.T X[j, :] y[j] } ]^2 
                     where , w ∈ ℝ^D , X ∈ ℝ^(SxD)

    ∂L(w)/∂w_q = w - C/(2S) Σ_{j=1, (w.T X[j,:] y[j]) <1}^S y[j] X[j,q] (1 - w.T X[j, :] y[j]) 
               = w - C/(2S) Σ_{j=1, (w.T X[j,:] y[j]) <1}^S y[j] X[j,q] (1 - Σ_{p=1}^D w_p X[j, p] y[j])
               
    """
    def __init__(self, n_epochs: int, batch_size: int, learning_rate: float, X: np.array, y: np.array,
                 offset: float, device_index: int, dc_index: int, type_of_cost: str, lambduh: float,
                 start_feature_idx: int = 0, decreasing_step=False, penalize_intercept=False, **kwargs):
        super(SVMCDClient, self).__init__(n_epochs, batch_size, learning_rate, X, y,
                                               offset, device_index, dc_index, type_of_cost, None,
                                               start_feature_idx, decreasing_step, penalize_intercept)
        self.C = kwargs["C"]

    def train(self, global_step_no, local_epochs=0):
        #logging.info(msg=f"Starting training for global level {global_step_no}")
        if local_epochs > 0:
            self.max_iter = local_epochs
        # Isolate the H_-k from other datacenters for the same label space
        # Obtained in the last iteration
        Xtheta_from_other_DC = self.Xtheta - self.X @ self.theta  # Assuming sample space is same
        for Q_idx, Q in enumerate(range(self.max_iter)):
            # batch gradient descent for the time being

            old_theta = copy.deepcopy(self.theta)
            h = Xtheta_from_other_DC + self.X @ self.theta
            max_diff = np.maximum(0, 1- np.multiply(self.y, h))
            if not self.contains_intercept:
                grad = self.theta - self.C*self.X.T @ np.multiply(self.y, max_diff)/len(self.y)
            else:
                w_without_intercept = copy.deepcopy(self.theta) 
                w_without_intercept[0] = 0
                grad = w_without_intercept - self.C*self.X.T @ np.multiply(self.y, max_diff)/len(self.y)
                
            self.theta = self.theta - self.alpha * grad

