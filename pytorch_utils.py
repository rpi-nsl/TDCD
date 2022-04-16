#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 23:15:11 2020

@author: Anirban Das
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List
import random
import copy
import os
from tqdm import tqdm
from PIL import Image
from print_metrics import print_metrics_binary
from sklearn.utils import shuffle


def normalize(x, means=None, stds=None):
    num_dims = x.shape[1]
    if means is None and stds is None:
        means = []
        stds = []
        for dim in range(num_dims):
            m = x[:, dim, :, :].mean()
            st = x[:, dim, :, :].std()
            x[:, dim, :, :] = (x[:, dim, :, :] - m)/st
            means.append(m.item())
            stds.append(st.item())
        return x , means, stds
    else:
        for dim in range(num_dims):
            m = means[dim]
            st = stds[dim]
            x[:, dim, :, :] = (x[:, dim, :, :] - m)/st
        return x , None, None
    
    
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        # to sent indices as well : https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/12
        return x, y, index

    def __len__(self):
        return self.tensors[0].size(0)
    

class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None, target_transform=None, perform_transform=False, datapoints=0, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.x = []
        self.y = []
        self.root = root
        self.x, self.y = shuffle(self.x,self.y, random_state=seed)

        self.classes, self.class_to_idx = self.find_classes(root)
        
        self.transform = transform
        self.target_transform = target_transform
        self.perform_transform = perform_transform
        self.datapoints = datapoints
        # root / <label>  / <train/test> / <item> / <view>.png
        for label in os.listdir(root): # Label
            for item in os.listdir(root + '/' + label + '/' + data_type):
                views = []
                for view in os.listdir(root + '/' + label + '/' + data_type + '/' + item):
                    views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)

                self.x.append(views)
                self.y.append(self.class_to_idx[label])
                
        if datapoints>0:
            self.x = self.x[:self.datapoints]
            self.y = self.y[:self.datapoints]
        
        if perform_transform:
            # perform the transform upfron instead of waiting for later
            self.x = self.transformDataset(self.x, self.transform)
        

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []
        if not self.perform_transform:
            for view in orginal_views:
                im = Image.open(view)
                im = im.convert('RGB')
                if self.transform is not None:
                    im = self.transform(im)
                views.append(im)
    
            return views, self.y[index], index
        else:
            # if the transform has already been performed
            return orginal_views, self.y[index], index
            
            
    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
    
    def transformDataset(self, data, transform):
        print("Transforming Dataset using ", transform)
        res = []
        for sample in tqdm(data):
            images = []
            for view in sample:
                im = Image.open(view)
                im = im.convert('RGB')
                im = transform(im)
                images.append(im)
            res.append(images)
        return res
    
    
class CifarNet(nn.Module):
    def __init__(self, ensemble=False):
        super(CifarNet, self).__init__()
        self.ensemble = ensemble
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5,3))
        self.fc1 = nn.Linear(128*5*2, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 128*5*2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

class CifarNetCombined(nn.Module):
    def __init__(self, ensemble=False, nb_classes=10, bias=False):
        super(CifarNetCombined, self).__init__()
        self.ensemble = ensemble
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5,3))
        self.fc1 = nn.Linear(128*5*2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, nb_classes, bias=bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 128*5*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

class CifarNetSimpleSmaller(nn.Module):
    def __init__(self, nb_classes=10, bias=False):
        # similar to https://www.tensorflow.org/tutorials/images/cnn
        super(CifarNetSimpleSmaller, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64,kernel_size=(3,3), padding=2)
        self.conv3 = nn.Conv2d(64, 64,kernel_size=(3,3))
        self.fc1 = nn.Linear(64 * 7 * 3, 64)
        self.fc2 = nn.Linear(64, nb_classes, bias=bias)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CifarNet2(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class MNIST_NET(nn.Module):
    def __init__(self, ensemble=False):
        super(MNIST_NET, self).__init__()
        self.ensemble = ensemble
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5,3))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5,3))
        self.fc1 = nn.Linear(64*4*2, 256)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64*4*2)
        x = self.fc1(x)
        return x 
    
    
class TopLayer(nn.Module):
    def __init__(self, linear_size=512, nb_classes=10, bias = False):
        super(TopLayer, self).__init__()
        self.classifier = nn.Linear(256+256, nb_classes, bias=bias)
        
    def forward(self, x):
        x = self.classifier(F.relu(x))
        return x
    
    
class MVCNN_NET(nn.Module):
    # from https://github.com/RBirkeland/MVCNN-PyTorch
    def __init__(self, num_classes=10):
        super(MVCNN_NET, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            #nn.ReLU(inplace=True),
            #nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        
        view_pool = []
        
        for v in x:
            v = self.features(v)
            v = v.view(v.size(0), 256 * 6 * 6)
            
            view_pool.append(v)
        
        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])
        
        pooled_view = self.classifier(pooled_view)
        return pooled_view    
    
    
class MVCNN_toplayer(nn.Module):
    def __init__(self, linear_size=4096, nb_classes=10, bias = False):
        super(MVCNN_toplayer, self).__init__()
        self.classifier = nn.Linear(1024+1024+1024+1024, nb_classes, bias=bias)
        
    def forward(self, x):
        x = self.classifier(F.relu(x))
        return x
    
class MIMICIII_LSTM_combined(nn.Module):
    # similar in structure to https://github.com/YerevaNN/mimic3-benchmarks
    # also keras lstm layers have three states but Pytorch lstm layers have an extra bias term for the input transformation and recurrent transformation
    # hence the total number of parameters are a bit different https://stackoverflow.com/a/48363392
    # https://stackoverflow.com/a/48725526
    def __init__(self, dim, input_dim, dropout=0.0, num_classes=1,
                 num_layers=1, batch_first=True, **kwargs):
        super(MIMICIII_LSTM_combined, self).__init__()
        self.hidden_dim = dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=self.input_dim, 
                            hidden_size=self.hidden_dim,
                            batch_first=batch_first)
        
        # self.initialize_weights(self.lstm)
        self.do = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hidden_dim, num_classes) # if self.hidden_dim=16 ,then we are effectively training a 64 layer output
        
    
    def forward(self, x):
        # Set initial states
        training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm.flatten_parameters()
        lstm_out2, (h2, c2) = self.lstm(self.do(x))
        # h2 and the last time step of lstm_out2 are equivalent
        h2 = h2.view(-1, self.hidden_dim) #== lstm_out2[:, -1, :]
        output = self.linear(h2)
        return output
        
        
    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

class MIMICIII_LSTM(nn.Module):
    # similar in structure to https://github.com/YerevaNN/mimic3-benchmarks
    # also keras lstm layers have three states but Pytorch lstm layers have an extra bias term for the input transformation and recurrent transformation
    # hence the total number of parameters are a bit different https://stackoverflow.com/a/48363392
    # https://stackoverflow.com/a/48725526
    def __init__(self, dim, input_dim, dropout=0.0, num_classes=1,
                 num_layers=1, batch_first=True, **kwargs):
        super(MIMICIII_LSTM, self).__init__()
        self.hidden_dim = dim
        self.input_dim = input_dim
        self.num_layers = num_layers
    
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.biLSTM = nn.LSTM(input_size=self.input_dim, 
                              hidden_size=self.hidden_dim//2, 
                              num_layers=1, 
                              bidirectional=True, 
                              batch_first=batch_first)
        
        self.lstm = nn.LSTM(input_size=self.hidden_dim, 
                            hidden_size=self.hidden_dim,
                            batch_first=batch_first)
        
        self.initialize_weights(self.biLSTM)
        self.initialize_weights(self.lstm)
        
        self.do = nn.Dropout(dropout)
        
    
    def forward(self, x):
        # Set initial states
        training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.biLSTM.flatten_parameters()
        h0 = torch.zeros(2, x.size(0), self.hidden_dim//2).to(training_device) # 2 for bidirection 
        c0 = torch.zeros(2, x.size(0), self.hidden_dim//2).to(training_device)
        lstm_out1, (h1, c1) = self.biLSTM(x, (h0, c0))
        # if you want to add the last hidden layers from both directions
        #hid_enc = torch.cat([h1[0,:, :], h1[1,:,:]], dim=1).unsqueeze(0)
        self.lstm.flatten_parameters()
        lstm_out2, (h2, c2) = self.lstm(self.do(lstm_out1))
        # h2 and the last time step of lstm_out2 are equivalent
        h2 = h2.view(-1, self.hidden_dim) #== lstm_out2[:, -1, :]
        output = h2
        return output
        
        
    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

class MIMICIII_lstm_toplayer(nn.Module):
    def __init__(self, linear_size=64, nb_classes=1, dropout= 0.0, bias = False):
        super(MIMICIII_lstm_toplayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(linear_size, nb_classes, bias=bias)
        
    def forward(self, x):
        x = self.classifier(self.dropout(x))
        return x
    
    
def add_model(dst_model, src_model):
    """Add the parameters of two models.
    Args:
        dst_model (torch.nn.Module): the model to which the src_model will be added.
        src_model (torch.nn.Module): the model to be added to dst_model.
    Returns:
        torch.nn.Module: the resulting model of the addition.
    """

    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data + dict_params2[name1].data)
    return dst_model


def scale_model(model, scale):
    """Scale the parameters of a model.
    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.
    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def federated_avg(models: Dict[Any, torch.nn.Module]) -> torch.nn.Module:
    """Calculate the federated average of a dictionary containing models.
       The models are extracted from the dictionary
       via the models.values() command.
    Args:
        models (Dict[Any, torch.nn.Module]): a dictionary of models
        for which the federated average is calculated.
    Returns:
        torch.nn.Module: the module with averaged parameters.
    """
    
    nr_models = len(models)
    model_list = list(models.values())
    device = torch.device('cuda' if next(model_list[0].parameters()).is_cuda else 'cpu')

    model = copy.deepcopy(model_list[0])
    model.to(device)
    # set all weights and biases of the model to 0
    model = scale_model(model, 0.0)

    for i in range(nr_models):
        model = add_model(model, model_list[i])
    model = scale_model(model, 1.0 / nr_models)
    return model


def get_train_or_test_loss(network_left,network_right,overall_top_layer,
                           overall_train_dataloader, 
                           overall_test_dataloader, report, cord_div_idx=16):
    network_left.eval()
    network_right.eval()
    overall_top_layer.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    # report with actual train set
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            
            output_left = network_left(data_left)
            output_right = network_right(data_right)

            input_top_layer = torch.cat((output_left, output_right), dim=1)
            output_top = overall_top_layer(input_top_layer)

            # test loss is the average loss of the two clients
            train_loss += F.cross_entropy(output_top, target.long()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.long().data.view_as(pred)).sum()
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset)))
    # report with actual test set
    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            
            output_left = network_left(data_left)
            output_right = network_right(data_right)

            input_top_layer = torch.cat((output_left, output_right), dim=1)
            output_top = overall_top_layer(input_top_layer)

            # test loss is the average loss of the two clients
            test_loss += F.cross_entropy(output_top, target.long()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.long().data.view_as(pred)).sum()
        test_loss /= len(overall_test_dataloader)
        
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        
        
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset)))


def general_get_train_or_test_loss_lstm(networks:List,
                                   overall_top_layer,
                                   overall_train_dataloader, 
                                   overall_test_dataloader, 
                                   report, coordinate_partitions=None):
    # the coordinate_partitions will contain the division of each time step features into coordinates
    num_parties = len(networks)
    for network in networks:
        network.eval()
    overall_top_layer.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    targets = []
    probabilities = []
    # report with actual train set
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:           
            vert_data = [data[:, :, coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)

            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]

            input_top_layer = torch.cat(H_embeddings, dim=1)
            output_top = overall_top_layer(input_top_layer)[:, 0]

            # test loss is the average loss of the two clients
            train_loss += F.binary_cross_entropy_with_logits(output_top, target.float()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.squeeze()>0.0#data.max(1, keepdim=True)[1]
            train_correct += pred.float().eq(target.float()).sum() #pred.eq(target.long().data.view_as(pred)).sum()
            
            targets.append(target.float().cpu())
            probabilities.append(torch.sigmoid(output_top.detach().cpu()))
            
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        probabilities = torch.cat(probabilities)    
        targets = torch.cat(targets)
        report["train_ret"].append(print_metrics_binary(targets, probabilities))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset)))
        
        
        
        
        
    targets = []
    probabilities = []
    # report with actual test set
    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            vert_data = [data[:, :,coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)
            
            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]

            input_top_layer = torch.cat(H_embeddings, dim=1)
            output_top = overall_top_layer(input_top_layer)[:, 0]

            # test loss is the average loss of the two clients
            test_loss += F.binary_cross_entropy_with_logits(output_top, target.float()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.squeeze()>0.0#.data.max(1, keepdim=True)[1]
            test_correct += pred.float().eq(target.float()).sum() #pred.eq(target.long().data.view_as(pred)).sum()
            
            targets.append(target.float().cpu())
            probabilities.append(torch.sigmoid(output_top.detach().cpu()))
            
        test_loss /= len(overall_test_dataloader)
        
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        probabilities = torch.cat(probabilities)    
        targets = torch.cat(targets)
        report["test_ret"].append(print_metrics_binary(targets, probabilities))
        
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset)))

def general_get_train_or_test_loss_lstm_combined(networks:List,
                                   overall_train_dataloader, 
                                   overall_test_dataloader, 
                                   report, coordinate_partitions=None):
    # the coordinate_partitions will contain the division of each time step features into coordinates
    num_parties = len(networks)
    for network in networks:
        network.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    targets = []
    probabilities = []
    # report with actual train set
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:           
            vert_data = [data[:, :, coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)

            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]

            input_top_layer = torch.cat(H_embeddings, dim=1)
            output_top = input_top_layer.sum(dim=1)

            # test loss is the average loss of the two clients
            train_loss += F.binary_cross_entropy_with_logits(output_top, target.float()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.squeeze()>0.0#data.max(1, keepdim=True)[1]
            train_correct += pred.float().eq(target.float()).sum() #pred.eq(target.long().data.view_as(pred)).sum()
            
            targets.append(target.float().cpu())
            probabilities.append(torch.sigmoid(output_top.detach().cpu()))
            
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        probabilities = torch.cat(probabilities)    
        targets = torch.cat(targets)
        report["train_ret"].append(print_metrics_binary(targets, probabilities))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset)))
        
        
        
        
        
    targets = []
    probabilities = []
    # report with actual test set
    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            vert_data = [data[:, :,coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)
            
            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]

            input_top_layer = torch.cat(H_embeddings, dim=1)
            output_top = input_top_layer.sum(dim=1)
            
            # test loss is the average loss of the two clients
            test_loss += F.binary_cross_entropy_with_logits(output_top, target.float()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.squeeze()>0.0#.data.max(1, keepdim=True)[1]
            test_correct += pred.float().eq(target.float()).sum() #pred.eq(target.long().data.view_as(pred)).sum()
            
            targets.append(target.float().cpu())
            probabilities.append(torch.sigmoid(output_top.detach().cpu()))
            
        test_loss /= len(overall_test_dataloader)
        
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        probabilities = torch.cat(probabilities)    
        targets = torch.cat(targets)
        report["test_ret"].append(print_metrics_binary(targets, probabilities))
        
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset)))
        
         
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0))
        return res
        
        
def get_train_or_test_loss_generic(networks:List,
                                   overall_train_dataloader, 
                                   overall_test_dataloader, 
                                   report, coordinate_partitions=None):
    # the coordinate_partitions will contain the division of each time step features into coordinates
    num_parties = len(networks)
    for network in networks:
        network.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    train_correct5 = 0
    test_correct5 = 0
    # report with actual train set
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:
            vert_data = [data[:, :, :, coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)
            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]
            #H_embeddings = []
            #for i in range(len(networks)):
            #    H_embeddings.append(networks[i](vert_data[i]))
            input_top_layer = torch.stack(H_embeddings)
            output_top = input_top_layer.sum(dim=0)

            #output_top = networks[0](vert_data[0])
            #for i in range(1,len(networks)):
            #    output_top += networks[i](vert_data[i])

            # test loss is the average loss of the two clients
            train_loss += F.cross_entropy(output_top, target.long()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.long().data.view_as(pred)).sum()
            train_correct5 += accuracy(output_top, target.long(), topk=(5,))[0]
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        report["train_accuracy5"].append(train_correct5 / len(overall_train_dataloader.dataset))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset),
            train_correct5 / len(overall_train_dataloader.dataset)))

    # report with actual test set
    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            vert_data = [data[:, :, :, coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)

            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]
            
            input_top_layer = torch.stack(H_embeddings)
            output_top = input_top_layer.sum(dim=0)
            
            # test loss is the average loss of the two clients
            test_loss += F.cross_entropy(output_top, target.long()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.long().data.view_as(pred)).sum()
            test_correct5 += accuracy(output_top, target.long(), topk=(5,))[0]
        test_loss /= len(overall_test_dataloader)
        
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        report["test_accuracy5"].append(test_correct5 / len(overall_test_dataloader.dataset))
        
        
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset),
            test_correct5 / len(overall_test_dataloader.dataset)))        
        
        
def get_train_or_test_loss_simplified_cifar(network_left,network_right,overall_train_dataloader, 
                           overall_test_dataloader, report, cord_div_idx=16):
    network_left.eval()
    network_right.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    # report with actual train set
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            
            output_left = network_left(data_left)
            output_right = network_right(data_right)

            output_top = output_right + output_left

            # test loss is the average loss of the two clients
            train_loss += F.cross_entropy(output_top, target.long()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.long().data.view_as(pred)).sum()
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset)))
    # report with actual test set
    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            
            output_left = network_left(data_left)
            output_right = network_right(data_right)

            output_top = output_right + output_left

            # test loss is the average loss of the two clients
            test_loss += F.cross_entropy(output_top, target.long()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.long().data.view_as(pred)).sum()
        test_loss /= len(overall_test_dataloader)
        
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        
        
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset)))


if __name__ == "__main__":
    one = nn.Conv2d(20,13, 3)
    two =nn.Conv2d(20,13, 3)
    three = nn.Conv2d(20,13, 3)
    bb = federated_avg({1:one, 2:two, 3:three})
    
    assert torch.isclose(bb.weight.data, (one.weight.data + two.weight.data + three.weight.data)/3.0).sum() == bb.weight.data.numel()
    
    

