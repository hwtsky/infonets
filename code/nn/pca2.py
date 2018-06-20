# pca2.py
"""
Created on Thu May 31 21:26:55 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
from copy import deepcopy
import torch as tc
from .pca import PCA
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params, copy_params_in, get_keys_vals, flip
from ..utils.methods import get_outsize_by_eigv

__all__ = [
    "PARAMS_PCA2",
    "PCA2",
]

PARAMS_PCA2 = OrderedDict()
PARAMS_PCA2['eps'] = 1e-6
PARAMS_PCA2['seed'] = 1
PARAMS_PCA2['batch_size'] = None#int(1e+5)
PARAMS_PCA2['denoise'] = -2 # -1, 0, 1, 2, 3
PARAMS_PCA2['energy_ratio'] = 0.99#0.01#
PARAMS_PCA2['refsize'] = None
PARAMS_PCA2['iszca'] = False
PARAMS_PCA2['iscent'] = True
PARAMS_PCA2['ismean'] = False
PARAMS_PCA2['isfull'] = False
PARAMS_PCA2['ridge'] = None
PARAMS_PCA2['return_output'] = True
PARAMS_PCA2['ith_cls'] =  0
PARAMS_PCA2['balance_factor'] = None # 0.5


class PCA2(PCA):

    def __init__(self, input=None, dtype=None,
                 name='PCA2', params=None, **kwargs):
        super(PCA2, self).__init__(input, dtype=dtype, name=name)
        p = PARAMS_PCA2.copy()
        self._parameters_train = copy_params_in(def_params=p,
                                                params=params, **kwargs)
        self._learning_method = 'PCA2'
        self.register_buffer('U0', None)
        self.register_buffer('E0', None)

    @staticmethod
    def pca2(input=None, params=None, **kwargs):
        assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
        p = copy_params(is_raised=False, params=params, **kwargs)
        keyslist = ['iszca', 'iscent', 'ismean', 'batch_size', 'denoise',
                    'energy_ratio', 'refsize', 'isfull', 'eps', 'seed',
                    'ridge', 'return_output','ith_cls','balance_factor']
        assignlist = [False, True, False, None, -1, 0.99,
                      None, False, 1e-6, 1, None, True, 0, None]
        [iszca, iscent, ismean, batch_size, denoise, energy_ratio, refsize, isfull,
         eps, seed, ridge, return_output,
         ith_cls, balance_factor] = get_keys_vals(keyslist, assignlist, params=p)
        assert 0<= ith_cls < input.get_num_cls()
        assert input.get_len(ith_cls) > 0
        if batch_size is None or batch_size < 0:
            batch_size = 0
        max_iter_samples = input.max_iter_samples
        cat_end = input.cat_end
        input.set_sample_iter(batch_size, True)
        shape = input.size()[1:]
        dim = input.dim()
        if dim > 2:
            input.view_(-1)
        N0 = input.get_len(ith_cls)
        N1 = len(input) - N0
        if ((balance_factor is None) or
            (balance_factor < 0.0) or (balance_factor > 1.0)):
            balance_factor = N0/(N0+N1)
        R = input.get_len_list()
        M = len(R)
        if balance_factor == 0.5:
            R = [0.5/((M-1)*i) for i in R]
            R[ith_cls] = 0.5/N0
        else:
            R = [(1.0-balance_factor)/N1]*M
            R[ith_cls] = balance_factor/N0
        cls_label = input.get_label(ith_cls)
#        print('pca2: cls_label = {}'.format(cls_label))
        mean = None
        input.reiter()
        for X, label in input:
            if iscent:
                X = X - X.mean(1, keepdim=True)
            if ismean and label is cls_label:# 
                i = input.get_index(label)
                if mean is None:
                    mean = R[i]*X.sum(0)
                else:
                    mean = mean + R[i]*X.sum(0)
        input.reiter()
        xx0 = tc.tensor([0.0])
        xx1 = tc.tensor([0.0])
        for X, label in input:
            i = input.get_index(label)
            if iscent:
                X = X - X.mean(1, keepdim=True)
            if ismean:
                X = X - mean
            if label is cls_label:
                xx0 = xx0 + R[i]*tc.mm(X.t(), X)#(1.0/N0)
            else:
                xx1 = xx1 + R[i]*tc.mm(X.t(), X)#(1.0/N1)
        X = None
        input.reiter()
        eps2 = eps*eps
        xx0 = (1.0/(tc.trace(xx0)))*xx0 #eps2+
        xx1 = (1.0/(tc.trace(xx1)))*xx1 #eps2+
#        if balance_factor == 0.5:
#            xx0 = (1.0/(M-1)/(tc.trace(xx0)))*xx0 #eps2+
#            xx1 = (1.0/(tc.trace(xx1)))*xx1#eps2+
#        else:
#            xx0 = (2.0*balance_factor/(tc.trace(xx0)))*xx0 #eps2+balance_factor/N0
#            xx1 = (2.0*(1.0-balance_factor)/(tc.trace(xx1)))*xx1#eps2+(1.0-balance_factor)/N1
        xx = xx0 + xx1
        tc.manual_seed(seed)
        E, U = tc.symeig(xx, eigenvectors=True)
        xx = xx1 = None
        K = len(E)
        if refsize is None:
            refsize = K
        E[E<=eps2] = eps2
        E = flip(E)
        U = flip(U,1)
        S = tc.sqrt(E)#+eps
        K0 = get_outsize_by_eigv(E, energy_ratio, denoise, refsize)
        if (denoise == 3):
            S[K0:] = S[K0-1]
            K0 = K
        elif ridge is not None:
            if ridge < 0.0:
                ridge = 0.0
            elif ridge > 1.0:
                ridge = 1.0
            lambda0 = ridge * S[0]
            S = S + lambda0
            K0 = K
        S[S<eps] = eps
        V = U[:,0:K0]*(1.0/(S[0:K0]))#+eps
        if K0 == 1:
            V = V.view(K, 1)
        if iszca:
            V = tc.mm(V, U[:,0:K0].t())
            if K0 == 1:
                V = V.view(K, 1)
        xx0 = V.t().mm(xx0).mm(V)
        tc.manual_seed(seed)
        E0, U0 = tc.symeig(xx0, eigenvectors=True)
#        ipdb.set_trace()
        E0 = flip(E0)
        U0 = flip(U0, 1)
        E0[E0<eps] = eps
        E0[E0>1.0-eps] = 1.0 - eps
#        print('PAC2: K={}, K0={}, E0.size={}, U0.size=({}, {})'.format(K, K0, E0.size(0), U0.size(0), U0.size(1)))
        V = V.mm(U0)
        if not isfull:
            U = U[:,:K0]
            if K0 == 1:
                U = U.view(K, 1)
            S = S[:K0]
        if ismean:
            bias = tc.mv(V.t(), -mean)
        else:
            bias = None
        if return_output:
            output = []
            input.set_sample_iter(0)
            for X, _ in input:
                if iscent:
                    X = X - X.mean(1, keepdim=True)
                if ismean:
                    output.append(tc.addmm(bias, X, V))#X.add(-mean.view(1,-1)).mm(V)
                else:
                    output.append(X.mm(V))
            output = OrderedDataset(output, cat_end=cat_end,
                                    max_iter_samples=max_iter_samples)
            output.cls_labels = deepcopy(input.cls_labels)
            input.set_sample_iter(max_iter_samples, cat_end)
            if dim > 2:
                input.view_(shape)
            return V.t(), bias, U, S, mean, U0, E0, output
        else:
            input.set_sample_iter(max_iter_samples, cat_end)
            if dim > 2:
                input.view_(shape)
            return V.t(), bias, U, S, mean, U0, E0

    def train_exe(self, input=None, params=None, **kwargs):
        input = self.input if input is None else input
        if input is None:
            raise ValueError("{}.train_exe(input): input is None".format(self.name))
        else:
            assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
            self._parameters_train = copy_params_in(
                def_params=self._parameters_train, params=params, **kwargs)
            p = self._parameters_train.copy()
            out = self.pca2(input, params=p)
            weight = self.set_weight(out[0], False)
            bias = self.set_bias(out[1], False)
            self.register_buffer('U', out[2])
            self.register_buffer('S', out[3])
            self.register_buffer('mean', out[4])
            self.register_buffer('U0', out[5])
            self.register_buffer('E0', out[6])
            return (weight, bias)+ out[2:]

