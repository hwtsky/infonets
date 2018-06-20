# srelu.py
"""
Created on Tue May 29 16:32:41 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import srelu_grad
from . import update

__all__ = [
    "PARAMS_SRELU",
    "srelu",
]

PARAMS_SRELU = OrderedDict()
PARAMS_SRELU['eps'] = 1e-6
PARAMS_SRELU['seed'] = -1
PARAMS_SRELU['max_epoches'] = 100
PARAMS_SRELU['lr'] = 0.4
PARAMS_SRELU['minlr'] = 1e-4
PARAMS_SRELU['tao'] = 0.9
PARAMS_SRELU['momentum_decay']= 0.9
PARAMS_SRELU['save_history'] = False#True #
PARAMS_SRELU['display'] = 50 #0 # 1, ... # None
PARAMS_SRELU['beta'] = 1.0
PARAMS_SRELU['bias_requires_grad'] = False
PARAMS_SRELU['isnorm'] = True
PARAMS_SRELU['isrow'] = False
PARAMS_SRELU['isorth'] = True
PARAMS_SRELU['orthNum'] =  None#50 #
PARAMS_SRELU['margin'] =  None
PARAMS_SRELU['bias_start'] = None
PARAMS_SRELU['weight_end'] = None
PARAMS_SRELU['update_method'] = 'update' # 'update1' #


def srelu(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_SRELU,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isrow', 'isorth', 'orthNum', 
                'eps', 'seed', 'margin', 'update_method']
    assignlist = [True, False, True, None, 1e-6, 1, None, 'update']
    [isnorm, isrow, isorth, orthNum, eps, seed, 
     margin, update_method] = get_keys_vals(keyslist, assignlist, params=p)
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
    return f(srelu_grad, 'SReLUGrad', input, C, bias, p, margin)

