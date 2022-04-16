import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torchvision.transforms as transforms
from PIL import Image
import torch

"""
The ModelNet-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def load_xy(root, data_type, transform=None):
    x_data = []
    x_files = []
    y = []
    classes, class_to_idx = find_classes(root)

    # root / <label>  / <train/test> / <item> / <view>.png
    for label in os.listdir(root): # Label
        for item in os.listdir(root + '/' + label + '/' + data_type):
            original_views = []
            for view in os.listdir(root + '/' + label + '/' + data_type + '/' + item):
                original_views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)
            x_files.append(original_views)

            views = []
            for view in original_views:
                im = Image.open(view)
                im = im.convert('RGB')
                if transform is not None:
                    im = transform(im)
                views.append(im)

            x_data.append(views)
            x_data[-1] = torch.cat(x_data[-1], dim=2)
            y.append(class_to_idx[label])
    return x_data, x_files, y

def load_modelnet_10_data(data_dir):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform1 =transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    transform2 = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ])

    root = data_dir
    train_data, train_filenames, train_labels = load_xy(root, 'train', transform1)
    test_data, test_filenames, test_labels = load_xy(root, 'test', transform2)
    label_names, _ = find_classes(root)

    return train_data, train_filenames, train_labels, \
        test_data, test_filenames, test_labels, label_names


if __name__ == "__main__":
    """show it works"""

    modelnet_10_dir = "../../MVCNN_Pytorch/10class/classes/"#'modelnet-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_modelnet_10_data(modelnet_10_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.
