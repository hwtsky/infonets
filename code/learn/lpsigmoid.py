# lpsigmoid.py
"""
Created on Sun Jun  3 10:06:00 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import lpsigmoid_grad
from . import update

__all__ = [
    "PARAMS_LPSIGMOID",
    "lpsigmoid",
]

PARAMS_LPSIGMOID = OrderedDict()
PARAMS_LPSIGMOID['eps'] = 1e-6
PARAMS_LPSIGMOID['seed'] = -1
PARAMS_LPSIGMOID['max_epoches'] = 100
PARAMS_LPSIGMOID['lr'] = 0.4
PARAMS_LPSIGMOID['minlr'] = 1e-4
PARAMS_LPSIGMOID['tao'] = 0.9
PARAMS_LPSIGMOID['momentum_decay']= 0.9
PARAMS_LPSIGMOID['save_history'] = False#True #
PARAMS_LPSIGMOID['display'] = 50 #0 # 1, ... # None
PARAMS_LPSIGMOID['beta'] = 1.81
PARAMS_LPSIGMOID['bias_requires_grad'] = False
PARAMS_LPSIGMOID['isnorm'] = True
PARAMS_LPSIGMOID['isrow'] = False
PARAMS_LPSIGMOID['isorth'] = True
PARAMS_LPSIGMOID['orthNum'] = None#50 #
PARAMS_LPSIGMOID['alpha'] = 0.9 #0.0 <= alpha <= 1.0
PARAMS_LPSIGMOID['weight_end'] = None
PARAMS_LPSIGMOID['bias_start'] = None
PARAMS_LPSIGMOID['update_method'] = 'update' # 'update1' #


def lpsigmoid(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_LPSIGMOID,
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
    return f(lpsigmoid_grad, 'LPSigmoidGrad', input, C, bias, p, alpha)

