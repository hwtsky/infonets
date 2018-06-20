# test_svm.py
"""
Created on Thu Jun  7 23:21:53 2018

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
PARAMS['max_epoches'] = 250

PARAMS['denoise'] = -1 # -1, 0, 1, 2, 3
PARAMS['energy_ratio'] = 0.99#0.01#
PARAMS['refsize'] = None #300 #
PARAMS['iscent'] = True # False #
PARAMS['ismean'] = False #True # 

PARAMS['out_zoom'] = -1 # -1, 0
PARAMS['outsize'] = 1

PARAMS['bias_requires_grad'] = True # False # 
PARAMS['balance_factor'] = None #0.5 #

PARAMS['beta'] = 3.0
PARAMS['margin'] = 0.2

PARAMS['E0'] = None #1e-10 #

PARAMS['orthNum'] = 50 # None # 
PARAMS['bias_start'] = 120 # None #
PARAMS['weight_end'] = 120 # None #

PARAMS['momentum_decay'] = 0.9
PARAMS['save_history'] = False # True # 
PARAMS['display'] = None #0 # 1, ... # None
PARAMS['tao'] = 0.9

PARAMS['save_inter_vals'] = False
PARAMS['save_bases_filters'] = True
PARAMS['one_all'] = False
PARAMS['one_one_double'] = True

#######################
PARAMS0 = OrderedDict()
PARAMS0['iscent'] = True # False #
PARAMS0['isstd'] = True # False #
PARAMS0['ismean'] = True #False # 
PARAMS0['isnorm'] = False #True # 

###############################################################################
if __name__ == "__main__":

#    os.system("cls") 
    tc.default_generator
#    tc.manual_seed(seed)
    
    M = 10
    rootdir1 = it.utils.get_rootdir()
    fname = it.utils.path_join(rootdir1, 'data', 'MNIST', 'new', 'train.pt')#'CIFAR10'
    ds = tc.load(fname)
    ds = it.utils.Ordered2DPatches(ds.cls_list[0:M], kernel_size=None,
                                   extraction_step=5, indent=0,
                                   max_patches=5e+4)
    ds.view_(-1)
    ds = it.nn.Linear.predata(ds, params=PARAMS0)
    
    learning_method = 'srelu2'
    F = it.nn.Conn2(input=ds, learning_method=learning_method, params=PARAMS) #

    weight, bias, beta, margin, history = F.train_exe()

    # %%
#    for dd, label in ds:
#        print('Size = {}, label = {}'.format(dd.shape, label))
#    ds.reiter()
    
#    s,e = F.get_out_index(0,0)
#    W =  F.bases# F.filters# F.layers[0].U.t()#
#    W = it.utils.sort_weights(W)
#    images = it.utils.make_grid_2d(W, padding=1,scale_each=True, pad_value=0)
#    it.utils.imshow(images)
#    it.utils.show1d2(history, title='Objective History', xlabel='Iteration', ylabel='Objective')

    fname = it.utils.path_join(rootdir1, 'data', 'MNIST', 'new', 'test.pt')#'CIFAR10'
    ds0 = tc.load(fname)
    ds0 = it.utils.Ordered2DPatches(ds0.cls_list[0:M], kernel_size=None,
                                   stride=5, indent=0,
                                   max_patches=1e+5)
    ds0.view_(-1)
    ds0 = it.nn.Linear.predata(ds0, params=PARAMS0)
    
    #%%
    print('learning_method = {}:'.format(learning_method))
    ismean = True
    F.classify(prestr='train', ismean=ismean)
    F.classify(input=ds0, prestr='test', ismean=ismean)
