#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Anirban Das, Timothy Castiglia
Note:
    To save time, evaluate every 10 iterations
"""

import copy
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, r2_score, roc_auc_score
from sklearn.datasets import load_svmlight_file
import sys
import os
import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing import Dict
from typing import Any
from load_cifar_10 import load_cifar_10_data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_utils import CustomTensorDataset, normalize, federated_avg, CifarNetSimpleSmaller, get_train_or_test_loss_simplified_cifar
from models.resnet2 import *

# global_seed = 42
# torch.manual_seed(global_seed)
# random.seed(global_seed)
# np.random.seed(global_seed)
EPSILON = 0.0000001

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# seed_torch()

def log_time(file, string=""):
    if string == "" :
        with open(file, 'a') as f:
            f.write(f"Started at :{timer()} \n")
    else:
        with open(file, 'a') as f:
            f.write(f"Finished {string} :{timer()} \n")

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            print("here")
            
            iterator = iter(iterable)       
       
def sampleQqminibatches(Q, BATCH_SIZE, GLOBAL_INDICES, with_replacement=True, journal=True):
    """
    Pick Q minibatches via sampling with/without replacement for the ICASSP version
    OR
    Pick 1 minibatches via sampling with/without replacement for Q rounds of local iteration for the Journal version
    """
    if not journal:
        minibatches = []
        if with_replacement:    
            for i in range(Q):
                minibatches.append(random.sample(GLOBAL_INDICES, BATCH_SIZE))
        else:
            # shuffle the data indices, 
            # start from 0 and take the first Q minibatches
            copy_GLOBAL_INDICES = copy.deepcopy(GLOBAL_INDICES)
            random.shuffle(copy_GLOBAL_INDICES)
            start = 0
            for i in range(Q):
                minibatches.append(copy_GLOBAL_INDICES[start: (start+1)*BATCH_SIZE])
                start+=1
    else:
        minibatches = []
        sampleonce = random.sample(GLOBAL_INDICES, BATCH_SIZE)
        for i in range(Q):
            minibatches.append(sampleonce)
                
    return minibatches 
        
class CD(object):
    def __init__(self, alpha: float , X, 
                 y , index: int, offset: int, device_list: list, average_network: nn.Module) -> None:
        self.alpha: float = alpha
        self.costs = []
        self.X = X
        self.y = y
        self.index = index
        self.device_list = device_list
        self.average_network = average_network
        
class Device(object):
    def __init__(self, network: nn.Module, alpha: float , X, 
                 y, device_index: int, dc_index: int, offset: int, 
                 indices : list, batch_size, transform=None, momentum=0, sampling_with_replacement=False) -> None:
        self.alpha: float = alpha
        self.momentum: float = momentum
        self.indices = indices
        self.batch_size = batch_size
        self.X = pd.DataFrame(X.reshape(X.shape[0], 3*32*16))
        self.y = pd.DataFrame(y)
        self.X.set_index(np.array(self.indices), inplace=True)
        self.y.set_index(np.array(self.indices), inplace=True)
        self.device_index = device_index
        self.dc_index = dc_index
        self.offset = offset
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=alpha,
                      momentum=self.momentum)
        self.lastlayer_Xtheta = torch.zeros((len(X), 256))
    
    def reset_optimizer(self):
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.alpha,
                      momentum=self.momentum)
    
    def getBatchFromIndices(self,indices, Qindex):
        ## use set intersection to select which data points from this client are selected
        # intersected_data_points = set(indices).intersection(set(self.indices.keys()))
        ## find these data points from X and y and return
        # intersected_data_points = [i-self.device_index*self.offset for i in intersected_data_points]
        # return self.X[intersected_data_points], self.y[intersected_data_points]
        current_batch_index = indices[Qindex]
        intersected_data_points = set(current_batch_index).intersection(set(self.indices))
        return self.X.loc[intersected_data_points, :], self.y.loc[intersected_data_points, :], list(intersected_data_points)
        
def parse_args():
    """
    Parse command line arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='TDCD CIFAR')
    # parser.add_argument('--data', type=int, nargs='?', default=0,
    #                         help='dataset to use in training.')
    # parser.add_argument('--model', type=int, nargs='?', default=0,
    #                         help='model to use in training.')
    parser.add_argument('--seed', type=int, nargs='?', default=42,
                            help='Random seed to be used.')
    parser.add_argument('--hubs', type=int, nargs='?', default=2,
                            help='Number of hubs in system (N).')
    parser.add_argument('--clients', type=int, nargs='?', default=10,
                            help='Number of workers per hub (K).')
    parser.add_argument('--gepochs', type=int, nargs='?', default=1000,
                            help='Number of global iterations to train for.')
    parser.add_argument('--Q', type=int, nargs='?', default=4,
                            help='Number of local iterations for client.')
    parser.add_argument('--batchsize', type=int, nargs='?', default=640,
                            help='Batch size to use in Mini-batch in each client in each hub per local iteration.')
    parser.add_argument('--lr', type=float, nargs='?', default=0.01,
                            help='Learning rate of gradient descent.')
    parser.add_argument('--evalafter', type=float, nargs='?', default=10,
                        help='Number of steps after which evaluation must be done.')
    parser.add_argument('--withreplacement', action='store_true',
                            help='If batches are to be picked with sampling with replacement.')
    parser.add_argument('--momentum', type=float, nargs='?', default=0,
                            help='Number of local iterations for client.')
    parser.add_argument('--lambduh', type=float, nargs='?', default=0.01,
                            help='Regularization coefficient.')
    parser.add_argument('--resultfolder', type=str, nargs='?', default="./data/results/journal",
                            help='Results Folder.')
    parser.add_argument('--evaluateateveryiteration', action='store_true',
                            help='If set, then we will evaluate every local round. Else we will evaluate every Q rounds.')
    parser.add_argument('--stepLR', action='store_true',
                            help='If set, then we will decrease LR in some steps. By default this is false and system uses initial LR.')
    
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    
    # Parse input arguments
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed_torch(args.seed)
    
    # Load the a9a dataset. In this case we will be using a constant intercept feature 
    cifar_10_dir = "./data/cifar10"#'cifar-10-batches-py'
    X_train, train_filenames, y_train, X_test, test_filenames, y_test, label_names = load_cifar_10_data(cifar_10_dir)

    X_train = torch.FloatTensor(X_train)/255.0 #scale all images by 255
    X_train = X_train.permute(0,3, 1,2) # to make it 50000, 3, 32, 32
    
    X_test = torch.FloatTensor(X_test)/255.0 #scale all images by 255
    X_test = X_test.permute(0, 3, 1, 2)# to make it 10000, 3, 32, 32
    
    """
    We need to standardize the tensor dataset
    We normalize by : image = (image - mean) / std
    in this case, for cifar10/mnist, we have 
    https://github.com/kuangliu/pytorch-cifar/issues/19
    https://github.com/kuangliu/pytorch-cifar/issues/16
    https://stackoverflow.com/questions/50710493/cifar-10-meaningless-normalization-values
    """
    #X_train, means, stds = normalize(X_train)
    means = torch.tensor([0.4914, 0.4822, 0.4465])
    stds = torch.tensor([0.247, 0.243, 0.261])
    X_train.sub_(means[None, :, None, None]).div_(stds[ None, :, None, None])
    X_test.sub_(means[None, :, None, None]).div_(stds[ None, :, None, None])

    # X_train, _, _ = normalize(X_train, means, stds)
    # X_test, _, _ = normalize(X_test, means, stds)
    
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    X_train_numpy = X_train.numpy()
    y_train_numpy = y_train.numpy()

    # 1.a Now we assume that there are N DCs
    N = args.hubs 
    K = args.clients 
    global_epoch = args.gepochs 
    local_epoch = args.Q 
    local_batch_size = args.batchsize 
    coordinate_per_dc = int(X_train.shape[2]/N)
    extradatapointsinfirstdevice = X_train.shape[2] - coordinate_per_dc*N
    datapoints_per_device = int(X_train.shape[0]/(K))
    alpha = args.lr # 0.1 
    momentum = args.momentum
    lambduh = args.lambduh
    decreasing_step = False
    ###########################################################################
    #--------------------DATA DISTRIBUTION FOR EXPERIMENTS--------------------#
    ###########################################################################
    
    # 1.b create N DCs and distribute the coordinates between them
    dc_list = []
    global_weights = np.zeros((X_train.shape[2], 1))
    global_indices = list(range(len(X_train)))
    GLOBAL_INDICES = list(range(len(X_train)))

    """
     Divide coordinates in such a way that the if the number of coordinates is not divisible by N
     then distribute the extra coordinates from first partitions
     for e.g. if no. of coordinates = 76 and N=16, then first 12 hubs will have 5 features
     last 4 hubs will have 4 features
    """
    coordinate_partitions = []
    coordinate_per_dc = int(X_train.shape[2]/N)
    extradatapointsinfirstdevice = X_train.shape[2] - coordinate_per_dc*N
    i = 0
    while i< X_train.shape[2]:
        if extradatapointsinfirstdevice>0:
            coordinate_partitions.append(list(range(i, i+ coordinate_per_dc + 1)))
            extradatapointsinfirstdevice-=1
            i=i+coordinate_per_dc + 1
        else:
            coordinate_partitions.append(list(range(i, i+ coordinate_per_dc )))
            i=i+coordinate_per_dc
    
    
    training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    over_train_loader = torch.utils.data.DataLoader(CustomTensorDataset(tensors=(X_train, y_train), transform=None)
                                    , batch_size=1000, shuffle=False)
    over_test_loader = torch.utils.data.DataLoader(CustomTensorDataset(tensors=(X_test, y_test), transform=None)
                                    , batch_size=1000, shuffle=False)
    
    

    for i in range(N):
        coordinate_per_dc = len(coordinate_partitions[i])
        dc_X = X_train_numpy[:, :, :, coordinate_partitions[i]]
        # Create a list of device connected to each DC, and suppose all of them 
        # have same number of data points which we distribute from here
        device_list = []
        #network_local = CifarNetSimpleSmaller(nb_classes=10)
        network_local = ResNet18()

        for k in range(K):
            device_list.append(Device(alpha=alpha,
                                      momentum=momentum,
                                      X=dc_X[k*datapoints_per_device : (k+1) * datapoints_per_device, :, :, :],
                                      y=y_train_numpy[k*datapoints_per_device : (k+1) * datapoints_per_device],
                                      device_index=k,
                                      dc_index=i,
                                      offset=datapoints_per_device,
                                      indices = list(range(k*datapoints_per_device , (k+1) * datapoints_per_device)),
                                      batch_size = local_batch_size, 
                                      network = copy.deepcopy(network_local),
                                      sampling_with_replacement= args.withreplacement
                                ))
        
        # Create the Data Center and attach the list of devices to it
        dc_list.append(CD(alpha=alpha, # need very small alpha
                          X=dc_X,
                          y=y_train,
                          index=i,
                          offset=coordinate_per_dc, 
                          device_list=device_list,
                          average_network = copy.deepcopy(network_local)))
    
    del X_train, y_train
    
    
    ###########################################################################
    #--------------------------TRAINING---------------------------------------#
    ###########################################################################
    
    report = {"train_loss": [],
              "test_loss":[],
              "train_accuracy":[],
              "test_accuracy": [],
              "hyperparameters":args
              }
    START_EPOCH = 0
    
    PATH = (f"Checkpoint_Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr{alpha}_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}.pt")
    if os.path.exists(PATH):
        print(
            """
            --------------LOADING FROM CHECKPOINT-----------------
            """
        )
        checkpoint = torch.load(PATH)
        START_EPOCH = int(checkpoint['epoch']) + 1 # start from the next epoch
        for hub_idx in range(N):
            dc_list[hub_idx].average_network.load_state_dict(checkpoint["hub_average_network_state_dict"][hub_idx])
            for device_idx, device in enumerate(dc_list[hub_idx].device_list):
                device.network = copy.deepcopy(dc_list[hub_idx].average_network)
                device.reset_optimizer()

        if not args.stepLR:
            filename =f"Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr{alpha}_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}_eval{args.evaluateateveryiteration}_evalafter{args.evalafter}.pkl" 
        else:
            filename =f"Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr[{alpha},0.005,0.001]_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}_eval{args.evaluateateveryiteration}_evalafter{args.evalafter}.pkl" 
        
        f = open(os.path.join(args.resultfolder, filename), "rb")
        report = pickle.load(f)

    
    for t in range(START_EPOCH, global_epoch):
        
        print(f"Epoch {t}/{global_epoch}")
        """
        At the beginning of each epoch, we select Q mini batches for the next Q local steps.

        We then exchange the intermediate information between the DCs w.r.t. the minibatches
        
        At the end of the following loop, each DC will have 
            1. A average w_{I_k} where I_k corresponds to the cordinates it has
            2. It will have a [Xw]_{I_k} of dimension m x I_k, where the [Xw] 
                is formed by vertically stacking [Xw] from all devices in correct order
        """
        
        #mini_batch_indices = sampleQqminibatches(local_epoch, BATCH_SIZE, GLOBAL_INDICES, with_replacement=True)
        
        # Do training
        """
        Simulate as if 
            1. the devices selected in the batch does one forward prob to get the current embedding
            2. The Echange takes place such that the devices have current Xtheta from mirror devices of other hubs
            3. To simulate, we first pick the batches across all coresponding devices of hubs
        
        """
        """
        EXCHANGE PART
        0. Fix Q minibatches of indices
        1. Loop over all the data centers and get the current Xthetas from the clients and stack them. Ideally we should only get the Xthetas for the point in Q minibatches
            But this is a simulation, so we can get all of the points and then just filder out the rest.
        2. Sum all the Xthetas across the datacenters
        3. In each DC, filter our the Xtheta for the Q minibatches. Save them in list and send to the clients.
        """

        """
        STEP 1: 
        This is random sampling with replacement, but manually done. 
        For random sampling with replacement, see this : https://discuss.pytorch.org/t/drawing-a-single-random-batch-from-a-dataloader/31756/2
        """
        batch_for_round = {}
        batch_indices_and_exchange_info_for_epoch = {i:{} for i in range(N)}
        mini_batch_indices = sampleQqminibatches(local_epoch, args.batchsize, GLOBAL_INDICES, with_replacement=True)
        for k_idx, k in enumerate(range(N)):
            current_DC = dc_list[k_idx]
            otherhub_index = 1 if k_idx == 0 else 0
            for device_idx, device in enumerate(current_DC.device_list):
                batch_indices_and_exchange_info_for_epoch[k_idx][device.device_index] = []
                for iterations in range(local_epoch):
                    # Isolate the H_-k from other datacenters for the same label space
                    # Obtained in the last iteration
                    #temp_X , temp_y, batch_indices = next(device.dataloader)
                    temp_X , temp_y, batch_indices = device.getBatchFromIndices(mini_batch_indices, iterations)
                    # assert that the batch indices in this cand other hub are same by checking if the labels are equal or not

                    np.testing.assert_array_equal(np.array(dc_list[otherhub_index].device_list[device_idx].y.loc[batch_indices]),
                            np.array(device.y.loc[batch_indices]))
                    """
                    Perform a forward pass with the latest network values and store the end embeddings
                    """
                    device.network.to(training_device)
                    with torch.no_grad():
                        if len(temp_X)==0:
                            batch_indices_and_exchange_info_for_epoch[k_idx][device.device_index].append({"batch_indices": copy.deepcopy(batch_indices), "embedding":torch.zeros(1)})
                            continue
                        temp_X = torch.FloatTensor(np.array(temp_X).reshape(temp_X.shape[0], 3, 32, 16))
                        temp_X = temp_X.to(training_device)
                        output = device.network(temp_X)
                        batch_indices_and_exchange_info_for_epoch[k_idx][device.device_index]\
                            .append({"batch_indices": copy.deepcopy(batch_indices), "embedding":output})
                        # assert output_top.shape == torch.Size([len(batch_indices_and_exchange_info_for_epoch[k_idx][device.device_index]["batch_indices"]),10])
               
        """
        As there are only two hubs, step 2 and step 3 are incorporated in step 1
        """
                
        
        """
        DO THE ACTUAL TRAINING WITH THE ABOVE SELECTED BATCHES FOR THIS GLOBAL ROUND
        """            
        for iteration in range(local_epoch):      
            """
            Implement variable learning rate.
            between 0-8000 time steps it is 0.01
            between 8000-16000 time step it is 0.005
            between 16000-30000 time step it is 0.001
            """
            if args.stepLR:
                if t*local_epoch + iteration >= 8000 and t*local_epoch + iteration<16000:
                    print(f"\n\n LE {t*local_epoch + iteration} LR:0.005 ") 
                    for k_idx, k in enumerate(range(N)):
                        current_DC = dc_list[k_idx]
                        for device_idx, device in enumerate(current_DC.device_list):        
                            for g in device.optimizer.param_groups:
                                g['lr'] = 0.005
                elif t*local_epoch + iteration >= 16000:
                    print(f"\n\n LE {t*local_epoch + iteration} LR:0.001 ")
                    for k_idx, k in enumerate(range(N)):
                        current_DC = dc_list[k_idx]
                        for device_idx, device in enumerate(current_DC.device_list):        
                            for g in device.optimizer.param_groups:
                                g['lr'] = 0.001                 
            
            for hub_index, k in enumerate(range(N)):
                coordinate_per_dc = len(coordinate_partitions[k])
                current_DC = dc_list[hub_index]
                # now learn parallelly in each connected device in current_DC
                # Isolate the H_-k from other datacenters for the same label space
                # Obtained in the last iteration
                # start of local iterations
                """
                Since we are using the same minibatch for Q iterations for the journal
                """
                for device_idx, device in enumerate(current_DC.device_list):
                    # select the batch indices from the Q minibatches picked earlier
                    device.network.to(training_device)
                    device.network.train()
                    batch_indices = batch_indices_and_exchange_info_for_epoch[hub_index][device_idx][iteration]["batch_indices"]
                    temp_X , temp_y, _ = device.getBatchFromIndices(mini_batch_indices, iteration)
                    temp_X = torch.FloatTensor(np.array(temp_X).reshape(temp_X.shape[0], 3, 32, 16))
                    temp_y = torch.FloatTensor(np.array(temp_y))[:,0]
                    
                    if len(temp_X) ==0:
                        print(f"Client {device.device_index} of {device.dc_index} does not have any datapoints in {t}:{iteration}. \n Skipping this round of training.")
                        continue
                    
                    temp_X , temp_y = temp_X.to(training_device), temp_y.to(training_device)
                
                    device.optimizer.zero_grad()
                    
                    output = device.network(temp_X)
                    
                    if hub_index==0:
                        output_top_from_other_hub_client = batch_indices_and_exchange_info_for_epoch[1][device.device_index][iteration]["embedding"]
                    elif hub_index==1:    
                        output_top_from_other_hub_client = batch_indices_and_exchange_info_for_epoch[0][device.device_index][iteration]["embedding"]
                        
                    total_output = output+output_top_from_other_hub_client
                    loss = F.cross_entropy(total_output, temp_y.long())
                    loss.backward()
                    # Clear out the grads calculated for the other hub
                    device.optimizer.step()

            """
            After taking one local step in each device in each data center. We calculate the loss 
            """
            # GENERATE REPORT EVERY Q steps by averaging
            # Now generate report every 10 steps
            # if args.evaluateateveryiteration or (args.Q>1 and iteration==local_epoch-1) or (args.Q==1 and t%10==0):
            if args.evaluateateveryiteration or (t*local_epoch + iteration+1)%args.evalafter==0:
                print(f"calculating every {args.evalafter} rounds", local_epoch, iteration, t*local_epoch + iteration)
                averaged_networks = [None]*N
                for hub_index, k in enumerate(range(N)):
                    coordinate_per_dc = len(coordinate_partitions[k])
                    current_DC = dc_list[hub_index]
                    # Average weights for reporting but do not replace the local weights
                    per_batch_model_list = {}
                    # MPI_Reduce within each Data Center to average the model
                    for device_idx, device in enumerate(current_DC.device_list):
                        per_batch_model_list[device_idx] = copy.deepcopy(device.network) # as if the device sends the model to the DC
                        
                    # MPI_Reduce within each Data Center to average the model
                    averaged_networks[hub_index] = federated_avg(per_batch_model_list)
                
                """
                DCS exchange the top layer information between eachother without averaging, but concatenating, 
                This allows us to maintain a Oracle like overall top layer network
                """
                # This is the MPI reduce part between the DCs
                # Get train loss at eah local iteration for each global iteration
                get_train_or_test_loss_simplified_cifar(averaged_networks[0], averaged_networks[1], 
                                       over_train_loader, over_test_loader, report, cord_div_idx=16) # i.e.divide each image at 16th column

        
        ###########################################################
        # at the end of Q local iterations, average the device models at the hubs and send the back to the devices
        ###########################################################
        # MPI_Reduce within each Data Center to average the model
        for k_idx, k in enumerate(range(N)):
            current_DC = dc_list[k_idx]
            device_model_list = {}
            device_top_layer_model_list = {}
        
            for device_idx, device in enumerate(current_DC.device_list):
                device_model_list[device_idx] = copy.deepcopy(device.network) # as if the device sends the model to the DC
            
            # MPI_Reduce within each Data Center to average the model
            current_DC.average_network = federated_avg(device_model_list)
            # current_DC.theta[k_idx* coordinate_per_dc: (k_idx+1) * coordinate_per_dc] = current_DC.theta_average
            
            # MPI Scatter the average weights and then MPI Reduce the Xtheta from devices
            # MPI Scatter to distribute the model to the nodes corresponding to the current DC
            for device_idx, device in enumerate(current_DC.device_list):
                device.network = copy.deepcopy(current_DC.average_network)
                device.reset_optimizer()


            

        """
        Save Report and checkpoint
        """
        PATH = (f"Checkpoint_Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr{alpha}_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}.pt")
        torch.save({
            'epoch': t,
            'hub_average_network_state_dict' : [i.average_network.state_dict() for i in dc_list],
        }, PATH)
        # =============================================================================
        os.makedirs(f"{args.resultfolder}", exist_ok=True)
        if not args.stepLR:
            filename =f"Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr{alpha}_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}_eval{args.evaluateateveryiteration}_evalafter{args.evalafter}.pkl" 
        else:
            filename =f"Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr[{alpha},0.005,0.001]_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}_eval{args.evaluateateveryiteration}_evalafter{args.evalafter}.pkl" 
        f = open(os.path.join(args.resultfolder, filename), "wb")
        pickle.dump(report, f)
        # =============================================================================

