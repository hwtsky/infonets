# lkrelu.py
"""
Created on Tue May 29 22:41:33 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import lkrelu_grad
from . import update

__all__ = [
    "PARAMS_LKRELU",
    "lkrelu",
]

PARAMS_LKRELU = OrderedDict()
PARAMS_LKRELU['eps'] = 1e-3
PARAMS_LKRELU['seed'] = -1
PARAMS_LKRELU['max_epoches'] = 100
PARAMS_LKRELU['lr'] = 0.4
PARAMS_LKRELU['minlr'] = 1e-4
PARAMS_LKRELU['tao'] = 0.9
PARAMS_LKRELU['momentum_decay'] = 0.9
PARAMS_LKRELU['save_history'] = False#True #
PARAMS_LKRELU['display'] = 50 #0 # 1, ... # None
PARAMS_LKRELU['beta'] = 0.1 # 0 <= beta <=1
PARAMS_LKRELU['bias_requires_grad'] = False
PARAMS_LKRELU['isnorm'] = True
PARAMS_LKRELU['isrow'] = False
PARAMS_LKRELU['isorth'] = True
PARAMS_LKRELU['orthNum'] =  None#50 #
PARAMS_LKRELU['margin'] = -0.5
PARAMS_LKRELU['gap'] = 0.5 #gap >= 0
PARAMS_LKRELU['negative_slope'] = 0.1
PARAMS_LKRELU['weight_end'] = None
PARAMS_LKRELU['bias_start'] = None
PARAMS_LKRELU['update_method'] = 'update' # 'update1' #


def lkrelu(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_LKRELU,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isrow', 'isorth', 'orthNum', 'eps', 'seed', 
                'beta', 'margin', 'gap', 'negative_slope', 'update_method']
    assignlist = [True, False, True, None, 1e-3, -1, 
                  0.1, -0.5, 0.5, 0.1, 'update']
    [isnorm, isrow, isorth, orthNum, eps, seed, beta, margin, gap, 
     negative_slope, update_method] = get_keys_vals(keyslist, assignlist, params=p)
    assert 0 <= beta <=1
    assert 0 <= gap
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
    return f(lkrelu_grad, 'LKReLUGrad', 
                  input, C, bias, p, margin, gap, negative_slope)

