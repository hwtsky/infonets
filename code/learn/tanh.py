# tanh.py
"""
Created on Tue May 29 13:09:26 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import tanh_grad
from . import update

__all__ = [
    "PARAMS_TANH",
    "tanh",
]

PARAMS_TANH = OrderedDict()
PARAMS_TANH['eps'] = 1e-6
PARAMS_TANH['seed'] = 1
PARAMS_TANH['max_epoches'] = 100
PARAMS_TANH['lr'] = 0.4
PARAMS_TANH['minlr'] = 1e-4
PARAMS_TANH['tao'] = 0.9
PARAMS_TANH['momentum_decay']= 0.9
PARAMS_TANH['save_history'] = False#True #
PARAMS_TANH['display'] = 50 #0 # 1, ... # None
PARAMS_TANH['beta'] = 0.9
PARAMS_TANH['bias_requires_grad'] = False
PARAMS_TANH['isnorm'] = True
PARAMS_TANH['isrow'] = False
PARAMS_TANH['isorth'] = True
PARAMS_TANH['orthNum'] =  None#50 #
PARAMS_TANH['weight_end'] = None
PARAMS_TANH['bias_start'] = None
PARAMS_TANH['update_method'] = 'update' # 'update1' #

def tanh(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_TANH,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isrow', 'isorth', 'orthNum', 
                'eps', 'seed', 'update_method']
    assignlist = [True, False, True, None, 1e-6, 1, 'update']
    [isnorm, isrow, isorth, orthNum,
     eps, seed, update_method] = get_keys_vals(keyslist, assignlist, params=p)
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
    f = getattr(update, update_method, None)
    assert f is not None
    return f(tanh_grad, 'TanhGrad', input, C, bias, p)

