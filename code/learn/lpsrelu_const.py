# lpsrelu_const.py
"""
Created on Thu Jun  7 11:43:23 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import lpsrelu_const_grad
from . import update

__all__ = [
    "PARAMS_LPSRELU_CONST",
    "lpsrelu_const",
]

PARAMS_LPSRELU_CONST = OrderedDict()
PARAMS_LPSRELU_CONST['eps'] = 1e-6
PARAMS_LPSRELU_CONST['seed'] = -1
PARAMS_LPSRELU_CONST['max_epoches'] = 150
PARAMS_LPSRELU_CONST['lr'] = 0.4
PARAMS_LPSRELU_CONST['minlr'] = 1e-4
PARAMS_LPSRELU_CONST['tao'] = 0.9
PARAMS_LPSRELU_CONST['momentum_decay']= 0.9
PARAMS_LPSRELU_CONST['save_history'] = False#True #
PARAMS_LPSRELU_CONST['display'] = 50 #0 # 1, ... # None
PARAMS_LPSRELU_CONST['beta'] = 3.0
PARAMS_LPSRELU_CONST['beta0'] = 0.1
PARAMS_LPSRELU_CONST['bias_requires_grad'] = False
PARAMS_LPSRELU_CONST['isnorm'] = True
PARAMS_LPSRELU_CONST['isrow'] = False
PARAMS_LPSRELU_CONST['isorth'] = True
PARAMS_LPSRELU_CONST['orthNum'] =  None#50 #
PARAMS_LPSRELU_CONST['margin'] = None
PARAMS_LPSRELU_CONST['const_factor'] = 0.1 # 0<= const_factor <=1
PARAMS_LPSRELU_CONST['balance_factor'] = 0.5# 0<= const_factor <=1
PARAMS_LPSRELU_CONST['alpha'] = 0.9 #0.0 <= alpha <= 1.0
PARAMS_LPSRELU_CONST['weight_end'] = None
PARAMS_LPSRELU_CONST['bias_start'] = None
PARAMS_LPSRELU_CONST['const_fun'] = 'const_lptanh'
PARAMS_LPSRELU_CONST['update_method'] = 'update' # 'update1' #


def lpsrelu_const(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2, 'input.dim must be equal to 2'
    p = copy_params_in(def_params=PARAMS_LPSRELU_CONST,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isrow', 'isorth','orthNum', 'eps', 'seed', 'const_fun',
                'margin', 'beta0', 'const_factor','balance_factor', 'alpha', 'update_method']
    assignlist = [True, False, True, None, 1e-6, -1, 
                  'const_lptanh', None, 0.1, 0.1, 0.5, 0.9, 'update']
    [isnorm, isrow, isorth, orthNum, eps, seed, const_fun, margin, beta0, const_factor,
      balance_factor, alpha, update_method] = get_keys_vals(keyslist, assignlist, params=p)
    if (const_factor is None):
        const_factor = 0.1
    elif const_factor < 0.0:
        const_factor = 0.0
    elif const_factor > 1.0:
        const_factor = 1.0
    if ((balance_factor is None) or 
        (balance_factor < 0.0) or (balance_factor > 1.0)):
        balance_factor = 1.0/input.get_num_cls()
    if (balance_factor is None):
        balance_factor = 1.0/input.get_num_cls()
    elif balance_factor < 0.0:
        balance_factor = 0.0
    elif balance_factor > 1.0:
        balance_factor = 1.0
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
    
    cat_end = input.cat_end
    max_iter_samples = input.max_iter_samples
    input.set_sample_iter(0, True)
    center = []
    for X, _ in input:
        center.append(X.mean(0, keepdim=True))
    input.set_sample_iter(max_iter_samples, cat_end)
#    ipdb.set_trace()
    f = getattr(update, update_method, None)
    assert f is not None
    return f(lpsrelu_const_grad, 'LPSReLUConstGrad', input, C, bias, p, 
            const_fun, margin, beta0, const_factor, balance_factor, alpha, center)
