#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/16/20 9:54 PM 2020

@author: Anirban Das
"""
import numpy as np
from node.client_CD import LinearCDClient, LogisticCDClient, ClientWorkerCoordinateDescent


def getClientClass(clientType):
    if clientType == "linear":
        return LinearCDClient
    elif clientType == "logistic":
        return LogisticCDClient
    else:
        raise Exception()


class Hub(object):
    """
    The base class for a Hub. Each entity has one single hub.
    """
    def __init__(self, theta, alpha: float, Xtheta: np.array, X: np.array, y : np.array,
                 index: int, offset: int, device_list: list, clientType: str) -> None:
        self.alpha: float = alpha
        self.Xtheta: np.array = Xtheta
        self.costs = []
        self.theta = theta
        self.X = X
        self.y = y
        self.index = index
        self.theta_average = self.theta[index*offset : (index+1)*offset]
        self.local_estimate = None
        # From https://github.com/akaanirban/BlockCoordinateDescent/blob/c18622aeb8f08cba87d9ad47e4b1b252e93b213a/loss_functions.py#L78
        #self.lipschitz_constant = np.max(np.linalg.eig(X.T @ X)[0])
        self.device_list = device_list
        self.global_Xtheta=None
        self.client_class = getClientClass(clientType)

























