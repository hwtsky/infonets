# srelu2.py
"""
Created on Thu May 31 12:44:08 2018

@author: Wentao Huang
"""
import ipdb
from collections import OrderedDict
import torch as tc
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params_in, to_Param, get_keys_vals
from ..utils.methods import pre_weights, init_weights
from . import srelu2_grad
from . import update

__all__ = [
    "PARAMS_SRELU2",
    "srelu2",
]

PARAMS_SRELU2 = OrderedDict()
PARAMS_SRELU2['eps'] = 1e-6
PARAMS_SRELU2['seed'] = 1
PARAMS_SRELU2['max_epoches'] = 200
PARAMS_SRELU2['lr'] = 0.4
PARAMS_SRELU2['minlr'] = 1e-4
PARAMS_SRELU2['tao'] = 0.9
PARAMS_SRELU2['momentum_decay']= 0.9
PARAMS_SRELU2['save_history'] = False#True #
PARAMS_SRELU2['display'] = 50 #0 # 1, ... # None
PARAMS_SRELU2['beta'] = 3.0
PARAMS_SRELU2['bias_requires_grad'] = False # True #
PARAMS_SRELU2['isnorm'] = True
PARAMS_SRELU2['isrow'] = False
PARAMS_SRELU2['isorth'] = True
PARAMS_SRELU2['orthNum'] = 50 #None#
PARAMS_SRELU2['ith_cls'] = 0
PARAMS_SRELU2['balance_factor'] = None #0.5#
PARAMS_SRELU2['margin'] =  0.2
PARAMS_SRELU2['E0'] = None
PARAMS_SRELU2['bias_start'] = 100
PARAMS_SRELU2['weight_end'] = 100
PARAMS_SRELU2['update_method'] = 'update' # 'update1' #


def srelu2(input, C=None, bias=None, params=None, **kwargs):
    assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
    assert input.dim() == 2 and len(input) > 0, 'input.dim must be equal to 2 and len(input) > 0'
    p = copy_params_in(def_params=PARAMS_SRELU2,
                       params=params, **kwargs)
    keyslist = ['isnorm', 'isorth', 'isrow', 'eps', 'seed', 
                'ith_cls', 'balance_factor', 'margin', 'E0', 'update_method']
    assignlist = [True, True, False, 1e-6, 1, 0, 0.5, 0.0, None, 'update']
    [isnorm, isorth, isrow, eps, seed, ith_cls, balance_factor, 
     margin, E0, update_method] = get_keys_vals(keyslist, assignlist, params=p)
    assert 0<= ith_cls < input.get_num_cls()
    assert input.get_len(ith_cls) > 0
    if ((balance_factor is None) or 
        (balance_factor < 0.0) or (balance_factor > 1.0)):
        balance_factor = input.get_len(ith_cls)/len(input)
#    print('srelu2: ith_cls = {}'.format(ith_cls))
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
    return f(srelu2_grad, 'SReLU2Grad', input, C, bias, p, 
                  ith_cls, R, margin, E0)
