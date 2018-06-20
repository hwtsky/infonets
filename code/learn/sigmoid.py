# sigmoid.py
"""
Created on Sat May 19 21:46:00 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import sigmoid_grad
from . import update

__all__ = [
    "PARAMS_SIGMOID",
    "sigmoid",
]

PARAMS_SIGMOID = OrderedDict()
PARAMS_SIGMOID['eps'] = 1e-6
PARAMS_SIGMOID['seed'] = 1
PARAMS_SIGMOID['max_epoches'] = 100
PARAMS_SIGMOID['lr'] = 0.4
PARAMS_SIGMOID['minlr'] = 1e-4
PARAMS_SIGMOID['tao'] = 0.9
PARAMS_SIGMOID['momentum_decay']= 0.9
PARAMS_SIGMOID['save_history'] = False#True #
PARAMS_SIGMOID['display'] = 50 #0 # 1, ... # None
PARAMS_SIGMOID['beta'] = 1.81
PARAMS_SIGMOID['bias_requires_grad'] = False
PARAMS_SIGMOID['isnorm'] = True
PARAMS_SIGMOID['isrow'] = False
PARAMS_SIGMOID['isorth'] = True
PARAMS_SIGMOID['orthNum'] =  None#50 #
PARAMS_SIGMOID['weight_end'] = None
PARAMS_SIGMOID['bias_start'] = None
PARAMS_SIGMOID['update_method'] = 'update' # 'update1' #


def sigmoid(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_SIGMOID,
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
#    ipdb.set_trace()
    f = getattr(update, update_method, None)
    assert f is not None
    return f(sigmoid_grad, 'SigmoidGrad', input, C, bias, p)
