# test_conn.py
"""
Created on Sun Jun  3 10:19:05 2018

@author: Wentao Huang
"""
from collections import OrderedDict
import torch as tc
import infonets as it

PARAMS = OrderedDict()
PARAMS['eps'] = 1e-6
PARAMS['seed'] = 1
PARAMS['max_epoches'] = 100
PARAMS['batch_size'] = None # 1000 #
PARAMS['update_method'] = 'update' # 'update1' #

PARAMS['denoise'] = 1 # -2, -1, 0, 1, 2, 3
PARAMS['energy_ratio'] = 0.01#.99
PARAMS['iszca'] = False
PARAMS['iscent'] = True
PARAMS['ridge'] = None
PARAMS['refsize'] = 200 #None #

PARAMS['out_zoom'] = -1 # -1, 0, 1,
PARAMS['outsize'] = 144
PARAMS['outscale'] = 1

PARAMS['momentum_decay'] = 0.9
PARAMS['save_history'] = True #False#
PARAMS['display'] = 10 #0 # 1, ... # None
#PARAMS['beta'] = 1.81
PARAMS['bias_requires_grad'] = False

PARAMS['alpha'] =  0.1 #0.0 <= alpha <= 1.0

PARAMS['save_inter_vals'] = False
PARAMS['save_bases_filters'] = True

###############################################################################
if __name__ == "__main__":
    rootdir1 = it.utils.get_rootdir()

    fname = it.utils.path_join(rootdir1, 'data', 'MNIST', 'new', 'train.pt')#'CIFAR10'
    ds = tc.load(fname)
    
#    ds = it.utils.Ordered2DPatches(ds.cls_list[0:3], kernel_size=11, 
#                                   extraction_step=5, indent=0, 
#                                   max_patches=5e+4)
    ds.view_(-1)
    
#    ds.set_sample_iter(1000)

    F = it.nn.Conn(input=ds, learning_method='sigmoid', params=PARAMS) #sigmoid  tanh tanh_ov lpsigmoid  lptanh
    
    weight, bias, beta, margin, history = F.train_exe()

    # %%
#    for dd, label in ds:
#        print('Size = {}, label = {}'.format(dd.shape, label))
#    ds.reiter()

    W =  F.bases# F.filters# F.layers[0].U.t()#
    W = it.utils.sort_weights(W)
    images = it.utils.make_grid_2d(W, padding=1,scale_each=True, pad_value=0)
    it.utils.imshow(images)
    it.utils.show1d(history, title='Objective History', xlabel='Iteration', ylabel='Objective')


