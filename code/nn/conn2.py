# conn2.py
"""
Created on Sat Jun  2 12:31:54 2018

@author: Wentao Huang
"""
#import ipdb
from copy import deepcopy
import torch as tc
from . import fun
from .fun import fun_exe
from .conn import Conn
from .conn1 import Conn1, PARAMS_CONN1
from .pca2 import PCA2
from .base import ModuleList
from .learn import Learn
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params, copy_params_in, get_key_val
from ..utils.methods import get_filters2, get_bases2, set_diag_e


class Conn2(Conn1):

    def __init__(self, input=None, bias=None,
                 beta=None, margin=None,
                 learning_method=None,
                 dtype=None, name='Conn2',
                 params=None, **kwargs):
        super(Conn, self).__init__(in_features=None,
                                   out_features=None,
                                   bias=bias, weight=None, coff_a=None,
                                   beta=beta, margin=margin,
                                   requires_grad=True,
                                   dtype=dtype, name=name)
        self.input = input
        self._learning_method = learning_method
        p = self._parameters_train = PARAMS_CONN1.copy()
        self._parameters_train = copy_params_in(
                                 def_params=p, params=params, **kwargs)
#        ipdb.set_trace()
        layer0 = PCA2(dtype=dtype, params=params, **kwargs)
        layer1 = Learn(bias=bias, beta=beta, margin=margin,
                       learning_method=learning_method,
                       dtype=dtype, params=params, **kwargs)
        self.layers = ModuleList([layer0, layer1])
        E0 = get_key_val(key='E0', assign=None, 
                        is_raised=False, params=self.get_layer_param(1))
        self.set_parameters_train(E0=E0)
        self.register_parameter('weight', layer1.weight)
        self.register_parameter('bias', layer1.bias)
        self.register_parameter('margin', layer1.margin)
        self.register_parameter('beta', layer1.beta)
        self.register_parameter('coff_a', layer1.coff_a)
        self.register_buffer('bases', None)
        self.register_buffer('filters', None)
        self.register_buffer('outsize_list', None)
        self.register_buffer('history', None)
        self.register_buffer('e', None)

    def _train_exe2(self, input=None, params=None, **kwargs):
        if input is None:
            raise ValueError("{}._train_exe2(input): input is None".format(self.name))
        else:
            assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
            N, K = input.size()
            p = copy_params(is_raised=False, params=params, **kwargs)
            weight0, bias0, U, S, mean, U0, E0, output0 = self.layers[0].train_exe(#
                                        input, params=p, return_output=True)
#            output0 = self.layers[0].forward(input)
#            print('Conn2: K={}, E0.size={}, U0.size=({}, {})'.format(K, E0.size(0), U0.size(0), U0.size(1)))
#            E = get_key_val(key='E0', assign=None, is_raised=False, params=p)
#            if E is None:
            E = self.get_parameters_train(key='E0')
            if E is not None:
                E0 = E
#            ipdb.set_trace()
            KA, K = weight0.size()
            print('{}: K={}, KA={}, E0.size={}, E={}'.format(self.name, K, KA, E0.size(0), E))
            weight1, bias1, beta, margin, history = self.layers[1].train_exe(
                                                    output0, params=p, E0=E0)
            del output0
            eps = self.get_layer_param(1,'eps')
            beta = self.layers[1].beta
            C = beta*weight1.t()#
            e = set_diag_e(self.e, E, E0, C, eps)
            if e is not None and not tc.is_tensor(e):
                e = tc.tensor([e])
            self.e = e
            weight = weight1.mm(weight0)
            if mean is not None:    
                bias = tc.mv(weight, -mean)
            else:
                bias = None
            if bias1 is not None:
                if bias is None:
                    bias = bias1
                else:
                    bias = bias + bias1
            if self.get_parameters_train('save_bases_filters'):
                iszca = self.get_layer_param(0,'iszca')
                filters = get_filters2(weight1.t(), U, U0, iszca)
                bases = get_bases2(weight1.t(), U, S, U0, iszca)
            else:
                filters = tc.Tensor()
                bases = tc.Tensor()
            self.layers[1].history = None
            for i, layer in enumerate(self.layers):
                layer.empty_buffers()
                if i == 0:
                    layer.empty_parameters()
                else:
                    layer._parameters['weight'] = None
                    layer._parameters['bias'] = None
            return weight, bias, beta, margin, history, filters, bases

    def info(self, input=None):
        if input is None:
            input = self.input
        if input is None:
            return None
        lm = self.learning_method
        if '_const' in lm:
            lm = lm[0:-6]
        if '_ov' in lm:
            lm = lm[0:-3]
        if lm[-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            lm = lm[0:-1]
        lm = lm + '_grad'
        F = getattr(fun, lm, None)
        if F is None:
            raise ValueError('{}.info: No this function, {}'.format(
                              self.name, lm))
            
        alpha = self.get_layer_param(ith_layer=1, key='alpha')
        beta = self.beta
        self.set_beta(1.0)
        if alpha is None:
            output = fun_exe(fun, lm, self.forward(input), beta=beta)
        else:
            output = fun_exe(fun, lm, self.forward(input), alpha=alpha, beta=beta)
        self.set_beta(beta)
        eps = self.get_layer_param(1,'eps')
#        e = self.e
#        if e is not None:
#            eps = e
        if tc.is_tensor(output):
            return output.abs_().add_(eps).log_()
        elif isinstance(input, OrderedDataset):
            output0 = []
            max_iter_samples = output.max_iter_samples
            output.set_sample_iter(0)
            for X, name in output:
                output0.append(X.abs_().add_(eps).log_())
    #        ipdb.set_trace()
            output0 = OrderedDataset(output0, max_iter_samples=max_iter_samples)
            output0.cls_labels = deepcopy(output.cls_labels)
            return output0
        else:
            return None

