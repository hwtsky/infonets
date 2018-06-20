# lpsrelu2.py
"""
Created on Fri Jun  1 23:52:11 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import lpsrelu2_grad
from . import update

__all__ = [
    "PARAMS_LPSRELU2",
    "lpsrelu2",
]

PARAMS_LPSRELU2 = OrderedDict()
PARAMS_LPSRELU2['eps'] = 1e-6
PARAMS_LPSRELU2['seed'] = -1
PARAMS_LPSRELU2['max_epoches'] = 200
PARAMS_LPSRELU2['lr'] = 0.4
PARAMS_LPSRELU2['minlr'] = 1e-4
PARAMS_LPSRELU2['tao'] = 0.9
PARAMS_LPSRELU2['momentum_decay']= 0.9
PARAMS_LPSRELU2['save_history'] = False#True #
PARAMS_LPSRELU2['display'] = 50 #0 # 1, ... # None
PARAMS_LPSRELU2['beta'] = 3.0
PARAMS_LPSRELU2['bias_requires_grad'] = True
PARAMS_LPSRELU2['isnorm'] = True
PARAMS_LPSRELU2['isrow'] = False
PARAMS_LPSRELU2['isorth'] = True
PARAMS_LPSRELU2['orthNum'] = None#50 #
PARAMS_LPSRELU2['ith_cls'] = 0
PARAMS_LPSRELU2['balance_factor'] = 0.5
PARAMS_LPSRELU2['margin'] =  0.5
PARAMS_LPSRELU2['alpha'] = 0.9 #0.0 <= alpha <= 1.0
PARAMS_LPSRELU2['E0'] = None
PARAMS_LPSRELU2['weight_end'] = None
PARAMS_LPSRELU2['bias_start'] = None
PARAMS_LPSRELU2['update_method'] = 'update' # 'update1' #


def lpsrelu2(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2 and len(input) > 0, 'input.dim must be equal to 2 and len(input) > 0'
    p = copy_params_in(def_params=PARAMS_LPSRELU2,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isorth', 'isrow', 'eps', 'seed', 'ith_cls', 
                'balance_factor', 'margin', 'alpha', 'E0', 'update_method']
    assignlist = [True, True, False, 1e-6, 1, 0, 0.5, 0.0, 0.8, None, 'update']
    [isnorm, isorth, isrow, eps, seed, ith_cls, balance_factor, margin, alpha, 
     E0, update_method] = get_keys_vals(keyslist, assignlist, params=p)
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
    if E0 is not None:
        if tc.is_tensor(E0):
            if E0.dim() > 0:
                L = len(E0)
                assert L > 0
                E0[E0<eps] = eps
                E0[E0>1.0-eps] = 1.0 - eps
            else:
                if E0 <= eps:
                    E0 = eps
                elif E0 > 1.0 - eps:
                    E0 = 1.0 - eps
                E0 = tc.full((C.size(0),), E0)
#        else:
#            if E0 <= eps:
#                E0 = eps
#            elif E0 > 1.0 - eps:
#                E0 = 1.0 - eps
#            E0 = tc.full((C.size(0),), E0)
#    ipdb.set_trace()
    f = getattr(update, update_method, None)
    assert f is not None
    return f(lpsrelu2_grad, 'LPSReLU2Grad', 
                  input, C, bias, p, ith_cls, R, margin, alpha, E0)

