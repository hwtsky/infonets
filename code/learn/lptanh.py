# lptanh.py
"""
Created on Sun Jun  3 09:17:27 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import lptanh_grad
from . import update

__all__ = [
    "PARAMS_LPTANH",
    "lptanh",
]

PARAMS_LPTANH = OrderedDict()
PARAMS_LPTANH['eps'] = 1e-6
PARAMS_LPTANH['seed'] = -1
PARAMS_LPTANH['max_epoches'] = 100
PARAMS_LPTANH['lr'] = 0.4
PARAMS_LPTANH['minlr'] = 1e-4
PARAMS_LPTANH['tao'] = 0.9
PARAMS_LPTANH['momentum_decay']= 0.9
PARAMS_LPTANH['save_history'] = False#True #
PARAMS_LPTANH['display'] = 50 #0 # 1, ... # None
PARAMS_LPTANH['beta'] = 0.9
PARAMS_LPTANH['bias_requires_grad'] = False
PARAMS_LPTANH['isnorm'] = True
PARAMS_LPTANH['isrow'] = False
PARAMS_LPTANH['isorth'] = True
PARAMS_LPTANH['orthNum'] =  None#50 #
PARAMS_LPTANH['alpha'] =  0.9 #0.0 <= alpha <= 1.0
PARAMS_LPTANH['weight_end'] = None
PARAMS_LPTANH['bias_start'] = None
PARAMS_LPTANH['update_method'] = 'update' # 'update1' #


def lptanh(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_LPTANH,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isrow', 'isorth','orthNum', 
                'eps', 'seed', 'alpha', 'update_method']
    assignlist = [True, False, True, None, 1e-6, -1, 0.8, 'update']
    [isnorm, isrow, isorth, orthNum, eps, seed, 
     alpha, update_method] = get_keys_vals(keyslist, assignlist, params=p)
    assert 0.0<=alpha<=1.0, '0.0 <= alpha <= 1.0'
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
    return f(lptanh_grad, 'LPTanhGrad', input, C, bias, p, alpha)
