# test_predataset.py
"""
Created on Sun Jun  3 13:32:48 2018

@author: Wentao Huang
"""
import time
from collections import OrderedDict
#import matplotlib.pyplot as plt
#import torch as tc
import infonets as it

PARAMS = OrderedDict()
PARAMS['iscent'] = True # False #
PARAMS['isstd'] = True # False #
PARAMS['ismean'] = True #False # 
PARAMS['isnorm'] = False #True # 

###############################################################################
if __name__ == "__main__":
    
    for flag in [False]:#, True
        ds = it.misc.preMNIST(train=flag, issave=True, download=True, params=PARAMS)#
        for dd, label in ds:
            print('preMNIST: train = {}, Size = {}, label = {}'.format(flag, dd.shape, label))
        ds.reiter()
        print('')
        imgs, _ = ds.get_items_by_cls(flag, range(25))
        imgs = it.utils.make_grid(imgs, padding=1, scale_each=True, pad_value=1)
        it.utils.imshow(imgs)
        time.sleep(2)
        
    for flag in [False, True]:
        ds = it.misc.preCIFAR10(train=flag, issave=True, download=True, params=PARAMS)#
        for dd, label in ds:
            print('preCIFAR10: train = {}, Size = {}, label = {}'.format(flag, dd.shape, label))
        ds.reiter()
        print('')
        imgs, _ = ds.get_items_by_cls(flag, range(25))
        imgs = it.utils.make_grid(imgs, padding=1, scale_each=True, pad_value=0)
        it.utils.imshow(imgs)
        time.sleep(2)
        
    #%%    
    out = ds.apply(it.nn.fun, 'sigmoid_grad', 5.0)
#    out1 = it.nn.fun.fun_exe(None, 'sigmoid_grad', ds, 5.0)
    y, labels = out[:4]#ds[:4] #
    imgs = it.utils.make_grid(y, padding=1, scale_each=True, pad_value=0)
    it.utils.imshow(imgs)
    
    data, labels = ds.data()
    
    