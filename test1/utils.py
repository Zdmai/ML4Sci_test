# Reading/Writing Data
import os
import h5py as hp
import numpy as np
import urllib
import torch


from torch.utils.data import random_split


def download(url, filename):

    if filename not in os.listdir():
        urllib.request.urlretrieve(url, filename)


def train_valid_split(length, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * length)
    train_set_size = length - valid_set_size
    index = range(length)
    train_ind, valid_ind = random_split(index, \
    [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_ind), np.array(valid_ind)

def data_read(pa1, pa2):
    f1 = hp.File(pa1)
    x1 = np.array(f1['X'])
    y1 = np.array(f1['y'])

    f2 = hp.File(pa2)
    y2 = np.array(f2['y'])
    x2 = np.array(f2['X'])
    
    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0, dtype=np.float32)
    # X = X
    length = X.shape[0]

    return X, y, length

