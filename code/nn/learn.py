# learn.py
"""
Created on Sun May  6 19:14:36 2018

@author: Wentao Huang
"""
import ipdb
from collections import OrderedDict
from copy import deepcopy
import torch as tc
#from . import fun
#from .fun import fun_exe
from .linear import Linear
from ..utils.data import OrderedDataset
from ..utils.helper import to_Param, get_key_val, get_keys_vals, copy_params_ex, copy_params_in
from ..utils.methods import get_outsize, init_weights
from .. import learn

__all__ = [
    "PARAMS_LEARN",
    "Learn",
]

PARAMS_LEARN = OrderedDict()
PARAMS_LEARN['batch_size'] = None
PARAMS_LEARN['out_zoom'] = -1 # -1, 0, 1,
PARAMS_LEARN['outsize'] = 100
PARAMS_LEARN['outscale'] = 1 # 0, 1, 2, ...

LIST_LEARNING = ['sigmoid_uc', 'sigmoid_ov', 'tanh_uc', 'tanh_ov']

class Learn(Linear):

    def __init__(self, input=None, C=None, bias=None, 
                 beta=None, margin=None, learning_method=None,
                 dtype=None, name='Learn', params=None, **kwargs):
        weight = None if C is None else C.t()
        super(Learn, self).__init__(in_features=None,
                                    out_features=None,
                                    bias=bias, weight=weight, coff_a=None, 
                                    beta=beta, margin=margin,
                                    requires_grad=True,
                                    dtype=dtype, name=name)
        self._parameters_train = PARAMS_LEARN.copy()
        self.set_learning_method(learning_method, params=params, **kwargs)
        p = copy_params_ex(def_params=self._parameters_train, 
                           params=params, **kwargs)
        if self.weight is None:
            C = get_key_val(key='C', assign=None, params=p)
            if C is not None:
                self.set_weight(C.t(), dtype)
        if self.bias is None:
            bias = get_key_val(key='bias', assign=None, params=p)
            if bias is not None:
                self.set_bias(bias, dtype)
        if self.beta is None:
            beta = get_key_val(key='beta', assign=None, params=p)
            if beta is None:
                beta = 1.0
            self.set_beta(beta, dtype)
        if self.margin is None:
            margin = get_key_val(key='margin', assign=None, params=p)
            if margin is not None:
                self.set_margin(margin, dtype)
        if self.coff_a is None:
            coff_a = get_key_val(key='coff_a', assign=None, params=p)
            if coff_a is not None:
                self.set_coff_a(coff_a, False, dtype)
        self.register_buffer('history', None)
        self.input = input
    
    def set_learning_method(self, learning_method=None, params=None, **kwargs):
        self._learning_method = learning_method
        FParams = getattr(learn, 'PARAMS_'+self.learning_method.upper(), None)
        if learning_method is not None and FParams is None:
            raise ValueError('{}: No such learning_method, {}'.format(
                              self.name, self.learning_method))
        p = copy_params_ex(def_params=FParams, params=self._parameters_train)
        self._parameters_train = copy_params_in(def_params=p, 
                                                params=params, **kwargs)

    def train_exe(self, input=None, params=None, **kwargs):
        if self.learning_method is not None:
            input = self.input if input is None else input
            if input is None:
                raise ValueError("{}.train_exe(input): input is None".format(self.name))
            else:
                assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
#                ipdb.set_trace()
                F = getattr(learn, self.learning_method, None)
                if F is None:
                    raise ValueError('{}.train_exe: No this learning_method, {}'.format(
                                      self.name, self.learning_method))
                self._parameters_train = copy_params_in(
                    def_params=self._parameters_train, params=params, **kwargs)
                p = self._parameters_train.copy()
                weight, bias, margin, beta = self.weight, self.bias, self.margin, self.beta
                if beta is None:
                    beta = self.get_parameters_train(key='beta', assign=1.0)
                if margin is None:
                    margin = self.get_parameters_train(key='margin', assign=None)
                p['beta'] = beta
                p['margin'] = margin
                keyslist = ['batch_size', 'bias_requires_grad', 'out_zoom',
                            'outsize', 'outscale', 'seed']
                assignlist = [None, False, 0, 100, 1.0, 1]
                [batch_size, bias_requires_grad, out_zoom, outsize,
                 outscale, seed] = get_keys_vals(keyslist, assignlist, params=p)
                dim = input.dim()
                if dim > 2:
                    shape = input.size()[1:]
                    input = input.view_(-1)
                N, K = input.size()
                max_iter_samples = input.max_iter_samples
                cat_end = input.cat_end
                if batch_size is None:
                    batch_size = max_iter_samples
                elif batch_size < 0:
                    batch_size = 0
                input.set_sample_iter(batch_size, True)
                if weight is None:
#                    ipdb.set_trace()
                    KA = get_outsize(K, outsize, outscale, out_zoom)
                    tc.manual_seed(seed)
                    if seed < 0:
                        C = tc.eye(K, KA)
                        if K > KA:
                           C[KA:K,:] = init_weights(K-KA, KA, seed, 1.0).t()# tc.empty(K-KA, KA).uniform_(-1, 1)
                        elif K < KA:
                           C[:, K:KA] = init_weights(K, KA-K, seed, 1.0).t()# tc.empty(K, KA-K).uniform_(-1, 1)
                    else:
                        C = init_weights(K, KA, seed).t()
                    C = to_Param(C, True)
                else:
                    assert weight.dim() == 2 and K == weight.size(1)
                    C = weight.t()
                C, bias, history = F(input, C, bias, params=p)
                weight = self.set_weight(C.t(), True)
                bias = self.set_bias(bias, bias_requires_grad)
                beta = self.set_beta(beta)
                margin = self.set_margin(margin)
                save_history = self.get_parameters_train('save_history')
                if save_history:
                    self.register_buffer('history', history)
                input.set_sample_iter(max_iter_samples, cat_end)
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
        dim = input.dim()
        if isinstance(input, tc.Tensor):
            if dim == 2:
                return self.linear(input, weight, bias)
            else:
                shape = input.size()[1:]
                input = input.view(len(input), -1)
                output = self.linear(input, weight, bias)
                return output#.view((len(output),shape[0],-1))
        elif isinstance(input, OrderedDataset):
            if dim == 2:
                return self.ordlinear(input, weight, bias)
            else:
                shape = input.size()[1:]
                input = input.view_(-1)
                output = self.ordlinear(input, weight, bias)
                input.view_(shape)
                return output#.view((shape[0], -1))
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
        bias_requires_grad = self.get_parameters_train('bias_requires_grad')
        if bias_requires_grad:
            bias_requires_grad = 'bias_requires_grad=True, '
        else:
            bias_requires_grad = ''
        if tc.is_tensor(self.bias):
            bias = 'bias={:.2e}, '.format(self.bias.mean().item())
        else:
            bias = ''
        margin = 'margin={:.2f}, '.format(self.margin.item()) if self.margin is not None else ''
        beta = 'beta={:.2f}, '.format(self.beta.item()) if self.beta is not None else ''
        str_print0 = in_features+out_features+bias+beta+margin+bias_requires_grad
        str_print1 = 'learning_method={}, name={}'.format(
                      self.learning_method, self.name)
        return str_print0 + str_print1
