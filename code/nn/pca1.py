# pca1.py
"""
Created on Fri Jun  8 09:10:47 2018

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
    "PARAMS_PCA1",
    "PCA1",
]

PARAMS_PCA1 = OrderedDict()
PARAMS_PCA1['eps'] = 1e-6
PARAMS_PCA1['seed'] = 1
PARAMS_PCA1['batch_size'] = None#int(1e+5)
PARAMS_PCA1['denoise'] = -2 # -1, 0, 1, 2, 3
PARAMS_PCA1['energy_ratio'] = .99#0.01#
PARAMS_PCA1['refsize'] = None
PARAMS_PCA1['iszca'] = False
PARAMS_PCA1['iscent'] = True
PARAMS_PCA1['ismean'] = False
PARAMS_PCA1['isfull'] = False
PARAMS_PCA1['ridge'] = None
PARAMS_PCA1['return_output'] = True
PARAMS_PCA1['ith_cls'] =  0
PARAMS_PCA1['balance_factor'] = None# 0.5


class PCA1(PCA):

    def __init__(self, input=None, dtype=None,
                 name='PCA1', params=None, **kwargs):
        super(PCA1, self).__init__(input, dtype=dtype, name=name)
        p = PARAMS_PCA1.copy()
        self._parameters_train = copy_params_in(def_params=p,
                                                params=params, **kwargs)
        self._learning_method = 'PCA1'

    @staticmethod
    def pca1(input=None, params=None, **kwargs):
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
#        print('pca1: cls_label = {}'.format(cls_label))
#        cls_label = input.get_label(ith_cls)
#        print('pca1: iscent = {}, ismean = {}'.format(iscent, ismean))
        mean = None #tc.tensor([0.0])
        input.reiter()
        for X, label in input:
            if iscent:
                X = X - X.mean(1, keepdim=True)
            i = input.get_index(label)
            if ismean:#  and i == ith_cls
                if mean is None:
                    mean = R[i]*X.sum(0)
                else:
                    mean = mean + R[i]*X.sum(0)
        input.reiter()
        xx = tc.tensor([0.0])
        for X, label in input:
            i = input.get_index(label)
            if iscent:
                X = X - X.mean(1, keepdim=True)
            if ismean:
                X = X - mean
            xx = xx + R[i]*tc.mm(X.t(), X)
        X = None
        input.reiter()
        tc.manual_seed(seed)
        E, U = tc.symeig(xx, eigenvectors=True)
        xx = None #
        K = len(E)
        if refsize is None:
            refsize = K
        eps2 = eps*eps
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
        if not isfull:
            U = U[:,:K0]
            if K0 == 1:
                U = U.view(K, 1)
            S = S[:K0]
        if mean is not None:
            bias = tc.mv(V.t(), -mean)
        else:
            bias = None#tc.tensor([0.0])
        if return_output:
            output = []
            input.set_sample_iter(0)
            for X, _ in input:
                if iscent:
                    X = X - X.mean(1, keepdim=True)
                if ismean:
                    output.append(X.add(-mean.view(1,-1)).mm(V))#tc.addmm(bias, X, V)
                else:
                    output.append(X.mm(V))
            output = OrderedDataset(output, cat_end=cat_end,
                                    max_iter_samples=max_iter_samples)
            output.cls_labels = deepcopy(input.cls_labels)
            input.set_sample_iter(max_iter_samples, cat_end)
            if dim > 2:
                input.view_(shape)
            return V.t(), bias, U, S, mean, output
        else:
            input.set_sample_iter(max_iter_samples, cat_end)
            if dim > 2:
                input.view_(shape)
            return V.t(), bias, U, S, mean

    def train_exe(self, input=None, params=None, **kwargs):
        input = self.input if input is None else input
        if input is None:
            raise ValueError("{}.train_exe(input): input is None".format(self.name))
        else:
            assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
            self._parameters_train = copy_params_in(
                def_params=self._parameters_train, params=params, **kwargs)
            p = self._parameters_train.copy()
            out = self.pca1(input, params=p)
            weight = self.set_weight(out[0], False)
            bias = self.set_bias(out[1], False)
            self.register_buffer('U', out[2])
            self.register_buffer('S', out[3])
            self.register_buffer('mean', out[4])
            return (weight, bias)+ out[2:]

