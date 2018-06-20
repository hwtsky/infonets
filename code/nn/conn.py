# conn.py
"""
Created on Tue May 22 20:40:13 2018

@author: Wentao Huang
"""
import ipdb
from collections import OrderedDict
from copy import deepcopy
import torch as tc
from . import fun
from .fun import fun_exe
from .base import ModuleList
from .linear import Linear
from .learn import Learn
from .pca import PCA
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params, copy_params_in
from ..utils.methods import get_filters, get_bases

__all__ = [
    "PARAMS_CONN",
    "Conn",
]

PARAMS_CONN = OrderedDict()
PARAMS_CONN['save_inter_vals'] = False
PARAMS_CONN['save_bases_filters'] = False


class Conn(Linear):

    def __init__(self, input=None, bias=None,
                 beta=None, margin=None,
                 learning_method=None,
                 dtype=None, name='Conn',
                 params=None, **kwargs):
        super(Conn, self).__init__(in_features=None,
                                   out_features=None,
                                   bias=bias, weight=None, coff_a=None,
                                   beta=beta, margin=margin,
                                   requires_grad=True,
                                   dtype=dtype, name=name)
        self.input = input
        self._learning_method = learning_method
        p = self._parameters_train = PARAMS_CONN.copy()
        self._parameters_train = copy_params_in(
                                 def_params=p, params=params, **kwargs)
        layer0 = PCA(dtype=dtype, params=params, **kwargs)
        layer1 = Learn(bias=bias, beta=beta, margin=margin,
                       learning_method=learning_method,
                       dtype=dtype, params=params, **kwargs)
        self.layers = ModuleList([layer0, layer1])
        self.register_parameter('weight', layer1.weight)
        self.register_parameter('bias', layer1.bias)
        self.register_parameter('margin', layer1.margin)
        self.register_parameter('beta', layer1.beta)
        self.register_parameter('coff_a', layer1.coff_a)
        self.register_buffer('bases', None)
        self.register_buffer('filters', None)
        self.register_buffer('history', None)

    def set_layer_param(self, ith_layer=0, params=None, **kwargs):
        return self.layers[ith_layer].set_parameters_train(params=params, **kwargs)

    def get_layer_param(self, ith_layer=0, key=None):
        return self.layers[ith_layer].get_parameters_train(key=key)

    def get_history(self):
        return self.history

    def train_exe(self, input=None, params=None, **kwargs):
        if self.learning_method is not None:
            input = self.input if input is None else input
            if input is None:
                raise ValueError("{}.train_exe(input): input is None".format(self.name))
            else:
                assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
                dim = input.dim()
                if dim > 2:
                    shape = input.size()[1:]
                    input = input.view_(-1)
                N, K = input.size()
                p = copy_params(is_raised=False, params=params, **kwargs)
#                ipdb.set_trace()
                print('\nRuning {}: input.size=({}, {})'.format(self.name, N, K))
                weight0, bias0, U, S, mean, output0 = self.layers[0].train_exe(
                                            input, params=p, return_output=True)
                weight1, bias1, beta, margin, history = self.layers[1].train_exe(
                                              output0, params=p)
                del output0
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
                weight = self.set_weight(weight, True)
                bias = self.set_bias(bias, self.get_layer_param(1,'bias_requires_grad'))
                beta = self.set_beta(beta)
                margin = self.set_margin(margin)
#                iscent = self.get_layer_param(ith_layer=0, key='iscent')
#                self.set_parameters_train(iscent=iscent)
                save_history = self.get_layer_param(1,'save_history')
                if self.get_parameters_train('save_bases_filters'):
                    iszca = self.get_layer_param(0,'iszca')
                    filters = get_filters(weight1.t(), U, iszca)
                    bases = get_bases(weight1.t(), U, S, iszca)
                    self.register_buffer('bases', bases)
                    self.register_buffer('filters', filters)
                if save_history:
                    self.register_buffer('history', history)
                self.layers[1].history = None
                if not self.get_parameters_train('save_inter_vals'):
                    for i, layer in enumerate(self.layers):
                        layer.empty_buffers()
                        if i == 0:
                            layer.empty_parameters()
                        else:
                            layer._parameters['weight'] = None
                            layer._parameters['bias'] = None
                if dim > 2:
                    input.view_(shape)
                return weight, bias, beta, margin, history
        else:
            return None

    def forward(self, input=None):
        if input is None:
            input = self.input
        if input is None:
            return None
        if self.weight is None:
            raise ValueError('{}: _parameters[\'weight\'] is None'.format(self.name))
        weight, bias = self.get_weight_bias(self.weight, self.bias, self.beta, 
                                            self.margin, self.coff_a)
        iscent = self.get_layer_param(0, 'iscent')
        dim = input.dim()
        if isinstance(input, tc.Tensor):
            if dim == 2:
                return self.linear(input, weight, bias, iscent)
            else:
                shape = input.size()[1:]
                input = input.view(len(input), -1)
                output = self.linear(input, weight, bias, iscent)
                return output#.view((len(output),shape[0],-1))
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
        if tc.is_tensor(output):
            return output.abs_().add_(eps).log_()
        elif isinstance(input, OrderedDataset):
            output0 = []
            max_iter_samples = output.max_iter_samples
            output.set_sample_iter(0)
            for X, _ in output:
                output0.append(X.abs_().add_(eps).log_())
    #        ipdb.set_trace()
            output0 = OrderedDataset(output0, max_iter_samples=max_iter_samples)
            output0.cls_labels = deepcopy(output.cls_labels)
            return output0
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
#        if tc.is_tensor(self.bias):
#            bias = 'bias={:.2e}, '.format(self.bias.mean().item())
#        else:
#            bias = ''
#        margin = 'margin={:.2f}, '.format(self.margin.item()) if self.margin is not None else ''
#        beta = 'beta={:.2f}, '.format(self.beta.item()) if self.beta is not None else ''
        str_print0 = in_features + out_features #{}{}{}, bias, beta, margin
        str_print1 = 'learning_method={}, name={}'.format(
                      self.learning_method, self.name)
        return str_print0 + str_print1

