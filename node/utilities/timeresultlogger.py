#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:37:43 2020

@author: Anirban Das
"""

from timeit import default_timer as timer
import pickle
import os


class result_logger(object):
    def __init__(self, file):
        self.init_time = timer()
        self.last_time = timer()
        self.file = file

    def log_time(self, string=""):
        t = timer()
        if string == "":
            with open(self.file, 'a') as f:
                f.write(f"Started at :{t} \n")
        else:
            with open(self.file, 'a') as f:
                f.write(f"Finished {string} :{t} in : {t - self.last_time}\n")
        self.last_time = t

    def log_and_save_results(self, costs, filepath):
        self.log_time(string=filepath.split(os.sep)[-1])
        pickle.dump(costs, open(filepath, "wb"))
        self.last_time = timer()
