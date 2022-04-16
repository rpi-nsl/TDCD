#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:51:10 2020

@author: Anirban Das
"""

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(42)


class clientWorker(object):
    """
    Class for the clients which will do the training
    """

    def __init__(self, n_epochs: int, batch_size: int, learning_rate: float, X: np.array, y: np.array,
                 offset: float, model: nn.Module, client_index: int, dc_index: int):
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
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.offset = offset
        self.model = model
        self.client_index = client_index
        self.dc_index = dc_index
        # self.X = X#[self.client_index * self.offset:(self.client_index + 1) * self.offset]
        # self.y = y#[self.client_index * self.offset:(self.client_index + 1) * self.offset]
        self.dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        # self.dataloader = DataLoader(self.dataset, batch_size=len(X), shuffle=True)
        self.n_batches = int(np.ceil(len(X) / self.batch_size))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, n_epochs=0, device=""):
        """
        Create and train Neural Network on training data in in_dir
        Args:
            n_epochs: Number of epochs for training
            device: Whether training to be done in cpu or gpu

        Returns:
            self.model: Returns the trained model object: nn.Module
        """
        if n_epochs > 0:
            self.n_epochs = n_epochs
        self.model.train()
        self.model.to(device)
        for ep in range(self.n_epochs):
            # Shuffle the dataset after each epoch

            #  Run through each mini-batch
            #  for b in range(self.n_batches):
            for batch_X, batch_Y in self.dataloader:
                # zero out the grad of the optimizer
                self.optimizer.zero_grad()

                # Use CPU or GPU
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                #  Run the network on each data instance in the minibatch
                #  and then compute the object function value
                pred_Y = self.model(batch_X)
                loss = self.criterion(pred_Y, batch_Y)

                #  Back-propagate the gradient through the network using the
                #  implicitly defined backward function
                loss.backward()
                #   print(f"{self.dc_index, self.client_index, loss}")
                #  Complete the mini-batch by actually updating the parameters.
                self.optimizer.step()

        return self.model

    def set_model(self, global_model):
        """
        Replace the local model with the global model and reset the optimizer with the parameters of the new model

        Args:
            global_model: The global model as nn.Module object

        Returns: None

        """
        self.model = copy.deepcopy(global_model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
