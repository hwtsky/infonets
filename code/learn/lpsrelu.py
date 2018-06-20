# lpsrelu.py
"""
Created on Tue May 29 18:23:06 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import lpsrelu_grad
from . import update

__all__ = [
    "PARAMS_LPSRELU",
    "lpsrelu",
]

PARAMS_LPSRELU = OrderedDict()
PARAMS_LPSRELU['eps'] = 1e-6
PARAMS_LPSRELU['seed'] = 1
PARAMS_LPSRELU['max_epoches'] = 100
PARAMS_LPSRELU['lr'] = 0.4
PARAMS_LPSRELU['minlr'] = 1e-4
PARAMS_LPSRELU['tao'] = 0.9
PARAMS_LPSRELU['momentum_decay']= 0.9
PARAMS_LPSRELU['save_history'] = False#True #
PARAMS_LPSRELU['display'] = 50 #0 # 1, ... # None
PARAMS_LPSRELU['beta'] = 3.0
PARAMS_LPSRELU['bias_requires_grad'] = False
PARAMS_LPSRELU['isnorm'] = True
PARAMS_LPSRELU['isrow'] = False
PARAMS_LPSRELU['isorth'] = True
PARAMS_LPSRELU['orthNum'] =  None#50 #
PARAMS_LPSRELU['margin'] =  -1.0
PARAMS_LPSRELU['alpha'] =  0.9 #0.0 <= alpha <= 1.0
PARAMS_LPSRELU['weight_end'] = None
PARAMS_LPSRELU['bias_start'] = None
PARAMS_LPSRELU['update_method'] = 'update' # 'update1' #


def lpsrelu(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_LPSRELU,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isrow', 'isorth','orthNum', 'eps', 
                'seed', 'margin','alpha', 'update_method']
    assignlist = [True, False, True, None, 1e-6, 1, -1, 0.8, 'update']
    [isnorm, isrow, isorth, orthNum, eps, seed, margin, 
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
    return f(lpsrelu_grad, 'LPSReLUGrad', input, C, bias, p, margin, alpha)

