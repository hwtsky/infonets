# test_conv.py
"""
Created on Mon Jun 11 17:25:44 2018

@author: Wentao Huang
"""
import os
from collections import OrderedDict
import torch as tc
import infonets as it

seed = 1

PARAMS = OrderedDict()
PARAMS['eps'] = 1e-6
PARAMS['seed'] = seed
PARAMS['max_epoches'] = 100

PARAMS['iscent'] = True # False #
PARAMS['ismean'] = False #True # 
PARAMS['denoise'] = -2 # -2, -1, 0, 1, 2, 3
PARAMS['energy_ratio'] = 0.99#0.01#
PARAMS['refsize'] = 144 #None #

PARAMS['outsize'] = 144
PARAMS['out_zoom'] = -1 # -1, 0

PARAMS['bias_requires_grad'] = False # True # 
PARAMS['balance_factor'] = None #0.5 #

#PARAMS['beta'] = 1.81
#PARAMS['margin'] = None

PARAMS['orthNum'] = None # 50 # 
PARAMS['bias_start'] = 80 # None #
PARAMS['weight_end'] = 120 # None #

PARAMS['momentum_decay'] = 0.9
PARAMS['save_history'] = True # False # 
PARAMS['display'] = 10 #0 # 1, ... # None
PARAMS['tao'] = 0.9

##############
#PARAMS['beta'] = 1.0
PARAMS['margin'] = None
PARAMS['learning_method'] = 'tanh' # 'lpsrelu_const' # 'sigmoid' #
PARAMS['conn_method'] = 'Conn' # 'Conn', 'Conn1', 'Conn2'

kernel_size = 10

PARAMS['extraction_step'] = 5
PARAMS['indent'] = 0
PARAMS['max_patches'] = 2e+4
PARAMS['max_iter_samples'] = 0

PARAMS['save_inter_vals'] = False
PARAMS['save_bases_filters'] = True

#######################
PARAMS0 = OrderedDict()
PARAMS0['iscent'] = True # False #
PARAMS0['isstd'] = True # False #
PARAMS0['ismean'] = True #False # 
PARAMS0['isnorm'] = False #True # 

kwarg = dict(kernel_size=kernel_size, groups=1)

###############################################################################
if __name__ == "__main__":

    tc.default_generator
    
    M = 10
    rootdir1 = it.utils.get_rootdir()
    fname = it.utils.path_join(rootdir1, 'data', 'CIFAR10', 'new', 'test.pt')#'MNIST'
    ds = tc.load(fname)
    ds = it.utils.Ordered2DPatches(ds.cls_list[0:M], kernel_size=None,
                                   extraction_step=5, indent=0, max_patches=5e+4)
#    ds.view_(4, 14, 14)
#    ds = it.nn.Linear.predata(ds, params=PARAMS0)
 
    F = it.nn.Conv2d(input=ds, params=PARAMS, **kwarg) #kernel_size=kernel_size, groups=1

    weight, bias, beta, margin, history = F.train_exe()

    G = it.nn.MaxPool2d(2)
    
#    out = G(F())
#    config = it.utils.Config(a=1, b=2)
    
    # %%
#    for dd, label in ds:
#        print('Size = {}, label = {}'.format(dd.shape, label))
#    ds.reiter()
        
    W =  F.bases# F.filters# 
    W = it.utils.sort_weights(W)
    fun = 'make_grid_2d' if W.dim() == 2 else 'make_grid'
    grid = getattr(it.utils, fun, None)
    images = grid(W, padding=1, scale_each=True, pad_value=0)
    it.utils.imshow(images)
    it.utils.show1d(history, title='Objective History', 
                     xlabel='Iteration', ylabel='Objective')
    
    

