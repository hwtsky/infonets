# relu.py
"""
Created on Tue May 29 21:15:55 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import relu_grad
from . import update

__all__ = [
    "PARAMS_RELU",
    "relu",
]

PARAMS_RELU = OrderedDict()
PARAMS_RELU['eps'] = 1e-3
PARAMS_RELU['seed'] = -1
PARAMS_RELU['max_epoches'] = 100
PARAMS_RELU['lr'] = 0.4
PARAMS_RELU['minlr'] = 1e-4
PARAMS_RELU['tao'] = 0.9
PARAMS_RELU['momentum_decay'] = 0.9
PARAMS_RELU['save_history'] = False#True #
PARAMS_RELU['display'] = 50 #0 # 1, ... # None
PARAMS_RELU['beta'] = 1.0
PARAMS_RELU['bias_requires_grad'] = False
PARAMS_RELU['isnorm'] = True
PARAMS_RELU['isrow'] = False
PARAMS_RELU['isorth'] = True
PARAMS_RELU['orthNum'] =  None#50 #
PARAMS_RELU['margin'] = -0.5
PARAMS_RELU['weight_end'] = None
PARAMS_RELU['bias_start'] = None
PARAMS_RELU['update_method'] = 'update' # 'update1' #


def relu(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_RELU,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isrow', 'isorth', 'orthNum', 
                'eps', 'seed', 'margin', 'update_method']
    assignlist = [True, False, True, None, 1e-3, -1, -0.5, 'update']
    [isnorm, isrow, isorth, orthNum,
     eps, seed, margin, update_method] = get_keys_vals(keyslist, assignlist, params=p)
    Num, K0 = input.size()
    if C is None:
        if seed < 0:
            C = tc.eye(K0)
        else:
            C = init_weights(K0, K0, seed)
    assert C.dim() == 2, 'C.dim must be equal to 2'
    K, KA = C.size()
    assert K == K0 and K0 >= KA
    C = pre_weights(C, isorth, isnorm, isrow, eps)
    C = to_Param(C, True)
#    ipdb.set_trace()
    f = getattr(update, update_method, None)
    assert f is not None
    return f(relu_grad, 'ReLUGrad', input, C, bias, p, margin)
