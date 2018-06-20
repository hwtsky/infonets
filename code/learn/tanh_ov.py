# tanh_ov.py
"""
Created on Tue May 29 13:38:29 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import tanh_ov_grad
from . import update

__all__ = [
    "PARAMS_TANH_OV",
    "tanh_ov",
]

PARAMS_TANH_OV = OrderedDict()
PARAMS_TANH_OV['eps'] = 1e-6
PARAMS_TANH_OV['seed'] = 1
PARAMS_TANH_OV['max_epoches'] = 100
PARAMS_TANH_OV['lr'] = 0.4
PARAMS_TANH_OV['minlr'] = 1e-4
PARAMS_TANH_OV['tao'] = 0.9
PARAMS_TANH_OV['momentum_decay']= 0.8
PARAMS_TANH_OV['save_history'] = False#True #
PARAMS_TANH_OV['display'] = 50 #0 # 1, ... # None
PARAMS_TANH_OV['beta'] = 0.9
PARAMS_TANH_OV['bias_requires_grad'] = False
PARAMS_TANH_OV['isnorm'] = True
PARAMS_TANH_OV['isrow'] = True
PARAMS_TANH_OV['isorth'] = True
PARAMS_TANH_OV['orthNum'] =  None#50 #
PARAMS_TANH_OV['weight_end'] = None
PARAMS_TANH_OV['bias_start'] = None
PARAMS_TANH_OV['update_method'] = 'update' # 'update1' #


def tanh_ov(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_TANH_OV,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isrow', 'isorth', 'orthNum', 
                'eps', 'seed', 'update_method']
    assignlist = [True, True, True, None, 1e-6, 1, 'update']
    [isnorm, isrow, isorth, orthNum,
     eps, seed, update_method] = get_keys_vals(keyslist, assignlist, params=p)
    Num, K0 = input.size()
    isrow = True
    isnorm = True
    if C is None:
        if seed < 0:
            C = tc.eye(K0)
        else:
            C = init_weights(K0, K0, seed)
    assert C.dim() == 2, 'C.dim must be equal to 2'
    K, KA = C.size()
    assert K == K0 and K0 <= KA
    C = pre_weights(C, isorth, isnorm, isrow, eps)
    C = to_Param(C, True)
#    ipdb.set_trace()
    f = getattr(update, update_method, None)
    assert f is not None
    return f(tanh_ov_grad, 'TanhOVGrad', input, C, bias, p)

