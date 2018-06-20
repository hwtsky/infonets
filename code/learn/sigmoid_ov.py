# sigmoid_ov.py
"""
Created on Fri May 25 13:37:35 2018

@author:  Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import sigmoid_ov_grad
from . import update

__all__ = [
    "PARAMS_SIGMOID_OV",
    "sigmoid_ov",
]

PARAMS_SIGMOID_OV = OrderedDict()
PARAMS_SIGMOID_OV['eps'] = 1e-6
PARAMS_SIGMOID_OV['seed'] = 1
PARAMS_SIGMOID_OV['max_epoches'] = 100
PARAMS_SIGMOID_OV['lr'] = 0.4
PARAMS_SIGMOID_OV['minlr'] = 1e-4
PARAMS_SIGMOID_OV['tao'] = 0.9
PARAMS_SIGMOID_OV['momentum_decay']= 0.8
PARAMS_SIGMOID_OV['save_history'] = False#True #
PARAMS_SIGMOID_OV['display'] = 50 #0 # 1, ... # None
PARAMS_SIGMOID_OV['beta'] = 1.81
PARAMS_SIGMOID_OV['bias_requires_grad'] = False
PARAMS_SIGMOID_OV['isnorm'] = True
PARAMS_SIGMOID_OV['isrow'] = True
PARAMS_SIGMOID_OV['isorth'] = True
PARAMS_SIGMOID_OV['orthNum'] =  None#50 #
PARAMS_SIGMOID_OV['weight_end'] = None
PARAMS_SIGMOID_OV['bias_start'] = None
PARAMS_SIGMOID_OV['update_method'] = 'update' # 'update1' #


def sigmoid_ov(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_SIGMOID_OV,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isrow', 'isorth', 'orthNum', 
                'eps', 'seed', 'update_method']
    assignlist = [True, True, True, None, 1e-6, 1, 'update']
    [isnorm, isrow, isorth, orthNum,
     eps, seed, update_method] = get_keys_vals(keyslist, assignlist, params=p)
    isrow = True
    isnorm = True
    Num, K0 = input.size()
    if C is None:
        if seed < 0:
            C = tc.eye(K0)
        else:
            C = init_weights(K0, K0, seed)
    assert C.dim() == 2, 'C.dim must be equal to 2'
    K, KA = C.size()
    assert K0 == K and K0 <= KA
    C = pre_weights(C, isorth, isnorm, isrow, eps)
    C = to_Param(C, True)
#    ipdb.set_trace()
    f = getattr(update, update_method, None)
    assert f is not None
    return f(sigmoid_ov_grad, 'SigmoidOVGrad', input, C, bias, p)


