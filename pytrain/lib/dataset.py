import os, struct
from pytrain.lib import fs
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
import urllib2 as ul2
import os

def load_mnist(path=".", dataset="training", one_hot = False, digits=np.arange(10)):
    path = path + "/" + "mnist"
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels

def load_iris(path=".", dataset="training", one_hot = False):
    data_path = download_data(path, "iris")
    sample_data = os.path.join(data_path, "iris.csv")    
    dmat_train, dlabel_train, dmat_test, dlabel_test \
      = fs.csv_loader(sample_data, 0.2)
    if one_hot:
        one_hot_label = [[1,0,0], [0,1,0], [0,0,1]]
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
            ds = ul2.urlopen("https://github.com/socc-io/YamechineLearning/raw/master/sample_data/iris/iris.csv")
            with open(base_path + '/' + 'iris.csv', "wb") as ds_file:
                ds_file.write(ds.read())
        elif dataset_name == 'mnist':
            pass
        print("Download " + str(dataset_name) + " complete")
    else:
        print("Dataset " + str(base_path) + " is already exist")
    return base_path
