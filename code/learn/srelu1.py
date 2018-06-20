# srelu1.py
"""
Created on Fri Jun  1 01:33:17 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import srelu1_grad
from . import update

__all__ = [
    "PARAMS_SRELU1",
    "srelu1",
]

PARAMS_SRELU1 = OrderedDict()
PARAMS_SRELU1['eps'] = 1e-6
PARAMS_SRELU1['seed'] = 1
PARAMS_SRELU1['max_epoches'] = 200
PARAMS_SRELU1['lr'] = 0.4
PARAMS_SRELU1['minlr'] = 1e-4
PARAMS_SRELU1['tao'] = 0.9
PARAMS_SRELU1['momentum_decay']= 0.9
PARAMS_SRELU1['save_history'] = False#True #
PARAMS_SRELU1['display'] = 50 #0 # 1, ... # None
PARAMS_SRELU1['beta'] = 3.0
PARAMS_SRELU1['bias_requires_grad'] = True
PARAMS_SRELU1['isnorm'] = True
PARAMS_SRELU1['isrow'] = False
PARAMS_SRELU1['isorth'] = True
PARAMS_SRELU1['orthNum'] = 50 #None#
PARAMS_SRELU1['ith_cls'] = 0
PARAMS_SRELU1['balance_factor'] = 0.5
PARAMS_SRELU1['margin'] =  0.2
PARAMS_SRELU1['bias_start'] = 100
PARAMS_SRELU1['weight_end'] = 100
PARAMS_SRELU1['update_method'] = 'update' # 'update1' #

def srelu1(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2 and len(input) > 0, 'input.dim must be equal to 2 and len(input) > 0'
    p = copy_params_in(def_params=PARAMS_SRELU1,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isorth', 'isrow', 'eps', 'seed', 
                'ith_cls', 'balance_factor', 'margin', 'update_method']
    assignlist = [True, True, False, 1e-6, 1, 0, 0.5, 0.5, 'update']
    [isnorm, isorth, isrow, eps, seed, ith_cls, balance_factor, 
     margin, update_method] = get_keys_vals(keyslist, assignlist, params=p)
    assert 0<= ith_cls < input.get_num_cls()
    assert input.get_len(ith_cls) > 0
    if ((balance_factor is None) or
        (balance_factor < 0.0) or (balance_factor > 1.0)):
        balance_factor = input.get_len(ith_cls)/len(input)
    N0 = input.get_len(ith_cls)
    N1 = len(input) - N0
    R = input.get_len_list()
    M = len(R)
    if balance_factor == 0.5:
        R = [0.5/((M-1)*i) for i in R]
        R[ith_cls] = 0.5/N0
    else:
        R = [(1.0-balance_factor)/N1]*M
        R[ith_cls] = balance_factor/N0
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
    return f(srelu1_grad, 'SReLU1Grad',
                  input, C, bias, p, ith_cls, R, margin)

