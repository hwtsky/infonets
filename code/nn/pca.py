# pca.py
"""
Created on Tue May 15 16:40:35 2018

@author: Wentao Huang
"""
from collections import OrderedDict
from copy import deepcopy
import torch as tc
#from torch.nn import functional as F
from .linear import Linear
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params, copy_params_in, get_keys_vals, flip
from ..utils.methods import get_outsize_by_eigv

__all__ = [
    "PARAMS_PCA",
    "PCA",
]

PARAMS_PCA = OrderedDict()
PARAMS_PCA['eps'] = 1e-6
PARAMS_PCA['seed'] = 1
PARAMS_PCA['batch_size'] = None#int(1e+5)
PARAMS_PCA['denoise'] = -2 # -2, -1, 0, 1, 2, 3
PARAMS_PCA['energy_ratio'] = .99#0.01#
PARAMS_PCA['refsize'] = None
PARAMS_PCA['iszca'] = False
PARAMS_PCA['iscent'] = True
PARAMS_PCA['ismean'] = True
PARAMS_PCA['isfull'] = False
PARAMS_PCA['ridge'] = None
PARAMS_PCA['return_output'] = True


class PCA(Linear):

    def __init__(self, input=None, dtype=None,
                 name='PCA', params=None, **kwargs):
        super(PCA, self).__init__(in_features=None,
                                  out_features=None,
                                  bias=None, weight=None,
                                  coff_a=None, beta=None,
                                  margin=None, requires_grad=False,
                                  dtype=dtype, name=name)
        p = PARAMS_PCA.copy()
        self._parameters_train = copy_params_in(def_params=p,
                                                params=params, **kwargs)
        self._learning_method = 'PCA'
        self.register_buffer('U', None)
        self.register_buffer('S', None)
        self.register_buffer('mean', None)
        self.input = input

    @staticmethod
    def pca(input=None, params=None, **kwargs):
        assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
        p = copy_params(is_raised=False, params=params, **kwargs)
        keyslist = ['iszca', 'iscent', 'ismean', 'batch_size', 'denoise',
                    'energy_ratio', 'refsize', 'isfull', 'eps',
                    'seed', 'ridge', 'return_output']
        assignlist = [False, True, True, None, -1, 0.99,
                      None, False, 1e-6, 1, None, True]
        [iszca, iscent, ismean, batch_size, denoise, energy_ratio, refsize, isfull,
         eps, seed, ridge, return_output] = get_keys_vals(keyslist, assignlist, params=p)
        if batch_size is None or batch_size < 0:
            batch_size = 0
        max_iter_samples = input.max_iter_samples
        cat_end = input.cat_end
        input.set_sample_iter(batch_size, True)
        shape = input.size()[1:]
        dim = input.dim()
        if dim > 2:
            input.view_(-1)
        Num = len(input)
        mean = None
        input.reiter()
        for X, _ in input:
            if iscent:
                X = X - X.mean(1, keepdim=True)
            if ismean:
                if mean is None:
                    mean = 1.0/Num*X.sum(0)
                else:
                    mean = mean + 1.0/Num*X.sum(0)
        input.reiter()
        xx = tc.tensor([0.0])
        for X, _ in input:
            if iscent:
                X = X - X.mean(1, keepdim=True)
            if ismean:
                X = X - mean
            xx = xx + (1.0/Num)*tc.mm(X.t(), X)
        X = None
        input.reiter()
        tc.manual_seed(seed)
        E, U = tc.symeig(xx, eigenvectors=True)
        xx = None
        K = len(E)
        if refsize is None:
            refsize = K
        eps2 = eps*eps
        E[E<=eps2] = eps2
        E = flip(E)
        U = flip(U, 1)
        S = tc.sqrt(E)#+eps
        K0 = get_outsize_by_eigv(E, energy_ratio, denoise, refsize)
#        print('K0 = {}'.format(K0))
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
        if ismean:
            bias = tc.mv(V.t(), -mean)
        else:
            bias = None
        if return_output:
            output = []
            input.set_sample_iter(0, True)
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
            out = self.pca(input, params=p)
            weight = self.set_weight(out[0], False)
            bias = self.set_bias(out[1], False)
            self.register_buffer('U', out[2])
            self.register_buffer('S', out[3])
            self.register_buffer('mean', out[4])
            return (weight, bias)+ out[2:] # weight, bias, U, S, mean, output

    def forward(self, input=None):
        if input is None:
            input = self.input
        if input is None:
            return None
        weight = self.weight
        bias = self.bias
        if weight is None:
            raise ValueError('{}: _parameters[\'weight\'] is None'.format(self.name))
        dim = input.dim()
        iscent = self.get_parameters_train('iscent')
        if isinstance(input, tc.Tensor):
            if dim == 2:
                return self.linear(input, weight, bias)
            else:
                shape = input.size()[1:]
                input = input.view(len(input), -1)
                output = self.linear(input, weight, bias, iscent)
                return output.view((len(output),shape[0],-1))
        elif isinstance(input, OrderedDataset):
            if dim == 2:
                return self.ordlinear(input, weight, bias, iscent)
            else:
                shape = input.size()[1:]
                input = input.view_(-1)
                output = self.ordlinear(input, weight, bias, iscent)
                input.view_(shape)
                return output#.view_((shape[0], -1))
        else:
            return None

    def extra_repr(self):
        if self.in_features is None:
            in_features = ''
        else:
            in_features = 'in_features={}, '.format(self.in_features)
        if self.out_features is None:
            out_features = ''
        else:
            out_features = 'out_features={}, '.format(self.out_features)
        str_print0 = in_features + out_features
        str_print1 = 'learning_method={}, name={}'.format(
                                     self.learning_method, self.name)
        return str_print0 + str_print1


