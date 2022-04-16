import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.transforms import Bbox
from matplotlib.path import Path
from matplotlib.axes import Axes
from matplotlib.axes._axes import _make_inset_locator
from matplotlib.transforms import Bbox, Transform, IdentityTransform, Affine2D
from matplotlib.backend_bases import RendererBase
import matplotlib._image as _image
import numpy as np

import io
import torch

def getTimeonXaxisFromSteppedEvaluation(metric, comm, comp, Q, eval_after_steps):
    x = []
    sum_ = 0
    eval_after_steps = eval_after_steps
    # Now, total time for Q steps = X. 
    # Time needed for 1 local step = X/Q
    # Therefore time for eval_after_steps = X/Q*eval_after_steps
    total_time_for_Q_steps = 2 * comm + comm + Q*comp
    shortened_metric = [metric[i] for i in range(0, len(metric))]
    print("Time between each eval step" , total_time_for_Q_steps/Q*eval_after_steps)
    for i in range(len(shortened_metric)):
        x.append(sum_)
        # total time for each communication round
        # = 2 * comm time between server and clients + comm_time_between servers + Q * local iteration computation time
        # lets assume comm time betwen servers =  comm time between server and clients
        sum_ = sum_ + total_time_for_Q_steps/Q*eval_after_steps
    
    return x, shortened_metric

FOLDER = "../data/results/journal"
# Q 1 K 1
colors = plt.get_cmap("Dark2").colors
lr = 0.0001
seed = 2021
for seedidx, seed in enumerate([2021]):
    for iii, K_val in enumerate([50, 100]):    
        fig=plt.figure(figsize=(10,6), dpi= 100, facecolor='w', edgecolor='k')                
        for q, Q_val in enumerate([1, 10, 20, 30]):    
            one_f = open(os.path.join(FOLDER, f"Simplified_SMALLER_Cifar_model_BS2000_N2_K{K_val}_Q{Q_val}_lr{lr}_momentum0.0_seed2021_samplingFalse_evalFalse_evalafter20.0.pkl"),'rb')
            one = pickle.load(one_f)
            loss = np.array([acc.item() for acc in one['test_accuracy']])[:int(35000/20)]
            skipsteps=3 # i.e. print loss after every 60 rounds.
            loss = loss[::skipsteps]    
            timeline = np.arange(len(loss))
            plt.plot(timeline*20*skipsteps, loss, label=f"Q:{Q_val}", linewidth=2, c=colors[q])
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(f"Iterations", fontsize=18)
        plt.ylabel("Test Accuracy", fontsize=18)
        #plt.show()
        plt.savefig(f'cifar_testacc_K{K_val}_lr{lr}_seed{seed}.pdf')

for seedidx, seed in enumerate([2021]):
    for iii, K_val in enumerate([50, 100]):    
        fig=plt.figure(figsize=(10,6), dpi= 100, facecolor='w', edgecolor='k')                
        for q, Q_val in enumerate([1, 10, 20, 30]):    
            one_f = open(os.path.join(FOLDER, f"Simplified_SMALLER_Cifar_model_BS2000_N2_K{K_val}_Q{Q_val}_lr{lr}_momentum0.0_seed2021_samplingFalse_evalFalse_evalafter20.0.pkl"),'rb')
            one = pickle.load(one_f)
            loss = np.squeeze(one["train_loss"])[:int(35000/20)]
            skipsteps=3 # i.e. print loss after every 60 rounds.
            loss = loss[::skipsteps]    
            timeline = np.arange(len(loss))
            plt.plot(timeline*20*skipsteps, loss, label=f"Q:{Q_val}", linewidth=2, c=colors[q])
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(f"Iterations", fontsize=18)
        plt.ylabel("Training Loss", fontsize=18)
        #plt.show()
        plt.savefig(f'cifar_trainloss_K{K_val}_Q{Q_val}_seed{seed}.pdf')

comp = 1
comm = 100

for comm in [10, 100]:    
    for iii, K_val in enumerate([50, 100]):    
        fig=plt.figure(figsize=(10,6), dpi= 100, facecolor='w', edgecolor='k')                
        for q, Q_val in enumerate([1, 10, 20, 30]):    
            one_f = open(os.path.join(FOLDER, f"Simplified_SMALLER_Cifar_model_BS2000_N2_K{K_val}_Q{Q_val}_lr{lr}_momentum0.0_seed2021_samplingFalse_evalFalse_evalafter20.0.pkl"),'rb')
            one = pickle.load(one_f)
            metric = np.array([acc.item() for acc in one['test_accuracy']])
            x, y = getTimeonXaxisFromSteppedEvaluation(metric, comm, comp, Q=Q_val, eval_after_steps=20)
            skipsteps=3 # i.e. print loss after every 60 rounds.
            x = x[::skipsteps]    
            y = y[::skipsteps]
            plt.plot(x,y, label=f"Q:{Q_val}", linewidth=2, c=colors[q])
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(-1000, x[-1])
        plt.xlabel(f"Time Units", fontsize=18)
        plt.ylabel("Test Accuracy", fontsize=18)
        plt.savefig(f'cifar_testacc_latency{comm}_K{K_val}_Q{Q_val}_seed{seed}.pdf')
