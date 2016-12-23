import os, struct
from pytrain.lib import fs
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
import urllib2 as ul2
import os

def load_mnist(path=".", dataset="training", one_hot = False):
    data_path = download_data(path, "mnist")
    train_data = os.path.join(data_path, "MNIST_train.small.csv")
    dmat_train, dlabel_train = fs.csv_loader(train_data, 0)
    test_data = os.path.join(data_path, "MNIST_test.small.csv")
    dmat_test, dlabel_test = fs.csv_loader(test_data, 0)
    dmat_train = map(lambda row : map(float, row), dmat_train)
    dmat_test = map(lambda row : map(float, row), dmat_test)    
    if one_hot :
        one_hot_label = map(list,list(np.eye(10)))
        temp_dlabel_train = []
        temp_dlabel_test = []
        for l in dlabel_train:
            temp_dlabel_train.append(one_hot_label[int(l)])
        for l in dlabel_test:
            temp_dlabel_test.append(one_hot_label[int(l)])
        dlabel_train = temp_dlabel_train
        dlabel_test = temp_dlabel_test
    if dataset == "training":
        return dmat_train, dlabel_train
    elif dataset == "testing":
        return dmat_test, dlabel_test
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

def load_iris(path=".", dataset="training", one_hot = False):
    data_path = download_data(path, "iris")
    sample_data = os.path.join(data_path, "iris.csv")    
    dmat_train, dlabel_train, dmat_test, dlabel_test \
      = fs.csv_loader(sample_data, 0.2)
    dmat_train = map(lambda row : map(float, row), dmat_train)
    dmat_test = map(lambda row : map(float, row), dmat_test)    
    if one_hot:
        one_hot_label = map(list, list(np.eye(3)))
        temp_dlabel_train = []
        temp_dlabel_test = []
        for l in dlabel_train:
            temp_dlabel_train.append(one_hot_label[int(l)])
        for l in dlabel_test:
            temp_dlabel_test.append(one_hot_label[int(l)])
        dlabel_train = temp_dlabel_train
        dlabel_test = temp_dlabel_test
    if dataset == "training":
        return dmat_train, dlabel_train
    elif dataset == "testing":
        return dmat_test, dlabel_test
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

def download_data(path=".", dataset_name = ""):
    base_path = path + "/" + dataset_name
    if not os.path.exists(base_path):
        print("Creating " + str(base_path) + " directory")
        os.makedirs(base_path)    
        print("Downloading " + str(dataset_name) + " into " + str(base_path))
        if dataset_name == 'iris':
            ds = ul2.urlopen("https://raw.githubusercontent.com/becxer/pytrain/master/sample_data/iris/iris.csv")
            with open(base_path + '/' + 'iris.csv', "wb") as ds_file:
                ds_file.write(ds.read())
        elif dataset_name == 'mnist':
            ds = ul2.urlopen("https://raw.githubusercontent.com/becxer/pytrain/master/sample_data/mnist/MNIST_train.small.csv")
            with open(base_path + '/' + 'MNIST_train.csv', "wb") as ds_file:
                ds_file.write(ds.read())
            ds = ul2.urlopen("https://raw.githubusercontent.com/becxer/pytrain/master/sample_data/mnist/MNIST_test.small.csv")
            with open(base_path + '/' + 'MNIST_test.csv', "wb") as ds_file:
                ds_file.write(ds.read())        
        print("Download " + str(dataset_name) + " complete")
    else:
        print("Dataset " + str(base_path) + " is already exist")
    return base_path
