# predataset.py
"""
Created on Sun May 13 21:13:22 2018

@author: Wentao Huang
"""
from collections import OrderedDict
import torch as tc
#import torch.utils.data as data
#from torch.utils import data as D
#from torch.utils.data import DataLoader as DL
from torchvision import datasets
import torchvision.transforms as transforms
from ..utils.data import OrderedDataset
from ..utils.helper import get_rootdir, path_join
from ..nn import Linear

PARAMS = OrderedDict()
PARAMS['iscent'] = True # False #
PARAMS['isstd'] = True # False #
PARAMS['ismean'] = True #False # 
PARAMS['isnorm'] = False #True # 

def preMNIST(train=True, issave=True, download=True, params=None, **kwargs):
    rootdir = get_rootdir()
    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    path = path_join(rootdir, 'data', 'MNIST')    
    dataset = datasets.MNIST(path, train=train, download=download, transform=transform)
    ds = OrderedDataset(dataset)#, 10
    ds = Linear.predata(ds, params=params, **kwargs)
    if issave:
        if train:
            fn = 'train.pt'
        else:
            fn = 'test.pt'
        fname = path_join(rootdir, 'data', 'MNIST', 'new', fn)
        tc.save(ds, fname)
    return ds


def preCIFAR10(train=True, issave=True, download=True, params=None, **kwargs):
    rootdir = get_rootdir()
    transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#
    path = path_join(rootdir, 'data', 'CIFAR10')    
    dataset = datasets.CIFAR10(path, train=train, download=download, transform=transform)
    ds = OrderedDataset(dataset)#, 10
    ds = Linear.predata(ds, params=params, **kwargs)
    if issave:
        if train:
            fn = 'train.pt'
        else:
            fn = 'test.pt'
        fname = path_join(rootdir, 'data', 'CIFAR10', 'new', fn)
        tc.save(ds, fname)
    return ds
