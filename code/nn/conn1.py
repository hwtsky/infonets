# conn1.py
"""
Created on Fri Jun  8 12:09:55 2018

@author: Wentao Huang
"""
#import ipdb
from collections import OrderedDict
import torch as tc
from .conn import Conn
from .pca1 import PCA1
from .base import ModuleList
from .learn import Learn
from ..utils.data import OrderedDataset
from ..utils.helper import copy_params, copy_params_in
from ..utils.methods import get_filters, get_bases

__all__ = [
    "PARAMS_CONN1",
    "Conn1",
]

PARAMS_CONN1 = OrderedDict()
PARAMS_CONN1['save_inter_vals'] = False
PARAMS_CONN1['save_bases_filters'] = False
PARAMS_CONN1['one_all'] = False
PARAMS_CONN1['one_one_double'] = True

class Conn1(Conn):

    def __init__(self, input=None, bias=None,
                 beta=None, margin=None,
                 learning_method=None,
                 dtype=None, name='Conn1',
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
        layer0 = PCA1(dtype=dtype, params=params, **kwargs)
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
        self.register_buffer('outsize_list', None)
        self.register_buffer('history', None)

    def get_out_index(self, ith_row=None, jth_col=None):
        out = self.outsize_list
        if out is None:
            return None
        if ith_row is None:
            start = 0
            end = out.sum().item()
            return start, end
        assert ith_row >=0 and ith_row < out.size(0)
        if out.dim() == 1:
            out = out.cumsum(0)
        else:
            out0 = out.sum(dim=1).cumsum(0)
            if jth_col is not None:
                out0 = tc.cat((tc.tensor([0]), out0))
                col = out[ith_row, :].cumsum(0)
                assert jth_col >=0 and jth_col < col.size(0)
                col = tc.cat((tc.tensor([0]), col))
                start = out0[ith_row] + col[jth_col]
                end = out0[ith_row] + col[jth_col+1]
                return start.item(), end.item()
            else:
                out = out0
        out = tc.cat((tc.tensor([0]), out))
        start = out[ith_row]
        end = out[ith_row+1]
        return start.item(), end.item()

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
                bias_requires_grad = self.get_layer_param(1,'bias_requires_grad')
                save_history = self.get_layer_param(1,'save_history')
                save_bases_filters = self.get_parameters_train('save_bases_filters')
                save_inter_vals = self.get_parameters_train('save_inter_vals')
                one_all = self.get_parameters_train('one_all')
                if one_all:
                    [weight, bias, beta, margin, history, filters, bases,
                     outsize_list]=self._train_exe_one_all(input, params, **kwargs)
                else:
                    [weight, bias, beta, margin, history, filters, bases,
                     outsize_list]=self._train_exe_one_one(input, params, **kwargs)
                weight = self.set_weight(weight, True)
                bias = self.set_bias(bias, bias_requires_grad)
                beta = self.set_beta(beta)
                margin = self.set_margin(margin)
                self.register_buffer('outsize_list', tc.tensor(outsize_list))
                if save_bases_filters:
                    self.register_buffer('bases', bases)
                    self.register_buffer('filters', filters)
                if save_history:
                    self.register_buffer('history', history)
                self.layers[1].history = None
                if not save_inter_vals:
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

    def _train_exe2(self, input=None, params=None, **kwargs):
        if input is None:
            raise ValueError("{}._train_exe2(input): input is None".format(self.name))
        else:
            assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
            N, K = input.size()
            p = copy_params(is_raised=False, params=params, **kwargs)
            weight0, bias0, U, S, mean, output0 = self.layers[0].train_exe(#
                                         input, params=p, return_output=True)
#            output0 = self.layers[0].forward(input)
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
            if self.get_parameters_train('save_bases_filters'):
                iszca = self.get_layer_param(0,'iszca')
                filters = get_filters(weight1.t(), U, iszca)
                bases = get_bases(weight1.t(), U, S, iszca)
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

    def _train_exe_one_all(self, input=None, params=None, **kwargs):
        p = copy_params(is_raised=False, params=params, **kwargs)
        self._parameters_train = copy_params_in(
                                    def_params=self._parameters_train, params=p)
        bias1 = self.layers[1].bias
        bias_requires_grad = self.get_layer_param(1,'bias_requires_grad')
        save_history = self.get_layer_param(1,'save_history')
        save_bases_filters = self.get_parameters_train('save_bases_filters')
        one_all = self.get_parameters_train('one_all')
        num_cls = input.get_num_cls()
        N, K = input.size()
        print('\nRuning {}._train_exe_one_all: input.size = ({}, {}), '
              'number of classes = {}, one_all = {}'.format(
                      self.name, N, K, num_cls, one_all))
        outsize_list = []
        weight = tc.Tensor()
        bias = None
        filters = tc.Tensor()
        bases = tc.Tensor()
        history = None
        for i in range(num_cls):
            N, K = input.size()
            print('\n[{}]: input.size = ({}, {})'.format(i, N, K))
            self.layers[1].set_bias(bias1, bias_requires_grad)
            for layer in self.layers:
                layer.set_parameters_train(ith_cls=i)
            [weight0, bias0, beta0, margin0, history0,
             filters0, bases0] = self._train_exe2(input, params=p)
            K0 = weight0.size(0)
            outsize_list.append(K0)
            weight = tc.cat((weight, weight0),0)
            if bias0 is not None:
                bias0 = bias0*tc.Tensor(K0)
                if bias is None:
                    bias = bias0
                else:
                    bias = tc.cat((bias, bias0))
            if save_bases_filters:
                filters = tc.cat((filters, filters0),0)
                bases = tc.cat((bases, bases0),0)
            if save_history and history0 is not None:
                if history is None:
                    history = history0.view(1, -1)
                else:
                    history0 = history0.view(1, -1)
                    history = tc.cat((history, history0), 0)
            if i == 0:
                beta = beta0
                margin = margin0
        return weight, bias, beta, margin, history, filters, bases, outsize_list

    def _train_exe_one_one(self, input=None, params=None, **kwargs):
        p = copy_params(is_raised=False, params=params, **kwargs)
        self._parameters_train = copy_params_in(
                                    def_params=self._parameters_train, params=p)
        bias00 = self.layers[1].bias
        bias_requires_grad = self.get_layer_param(1,'bias_requires_grad')
        save_history = self.get_layer_param(1,'save_history')
        save_bases_filters = self.get_parameters_train('save_bases_filters')
        one_all = self.get_parameters_train('one_all')
        one_one_double = self.get_parameters_train('one_one_double')
        num_cls = input.get_num_cls()
        N, K = input.size()
        print('\nRuning {}._train_exe_one_one: input.size=({}, {}), '
              'num_cls={}, one_all={}'.format(self.name, N, K, num_cls, one_all))
        for layer in self.layers:
            layer.set_parameters_train(ith_cls=0)
        max_iter_samples = input.max_iter_samples
        cat_end = input.cat_end
        outsize_list = []
        weight = tc.Tensor()
        bias = None
        filters = tc.Tensor()
        bases = tc.Tensor()
        history = None
        beta = None
        margin = None
        for i in range(num_cls):
            start = 0 if one_one_double else i+1
            if start >= num_cls:
                break
            weight1 = tc.Tensor()
            bias1 = None
            filters1 = tc.Tensor()
            bases1 = tc.Tensor()
            outsize_list1 = []
            history1 = None
            print('\n[{}]: num_cls={}, start={}, one_one_double={}'
                  ''.format(i, num_cls, start, one_one_double), end='')
            for j in range(0, num_cls, 1):
                if not one_one_double:
                    if j < start:
                        outsize_list1.append(0)
                        continue
                if i == j:
                    outsize_list1.append(0)
                    continue
                input0 = [input.cls_list[i], input.cls_list[j]]
                input0 = OrderedDataset(input0, cat_end=cat_end,
                                        max_iter_samples=max_iter_samples)
                N, K = input0.size()
                print('\n[{}][{}]: num_cls={}, input0.size='
                      '({}, {})'.format(i, j, num_cls, N, K))
                self.layers[1].set_bias(bias00, bias_requires_grad)
                [weight0, bias0, beta0, margin0, history0,
                 filters0, bases0] = self._train_exe2(input0, params=p)
                K0 = weight0.size(0)
                outsize_list1.append(K0)
                weight1 = tc.cat((weight1, weight0),0)
                if bias0 is not None:
                    bias0 = bias0*tc.Tensor(K0)
                    if bias1 is None:
                        bias1 = bias0
                    else:
                        bias1 = tc.cat((bias1, bias0))
                if save_bases_filters:
                    filters1 = tc.cat((filters1, filters0),0)
                    bases1 = tc.cat((bases1, bases0),0)
                if save_history and history0 is not None:
                    if history1 is None:
                        history1 = history0.view(1, -1)
                    else:
                        history0 = history0.view(1, -1)
                        history1 = tc.cat((history1, history0), 0)
                if beta is None:
                    beta = beta0
                if margin is None:
                    margin = margin0
            outsize_list.append(outsize_list1)
            weight = tc.cat((weight, weight1),0)
            if bias1 is not None:
                if bias is None:
                    bias = bias1
                else:
                    bias = tc.cat((bias, bias1))
            if save_bases_filters:
                filters = tc.cat((filters, filters1),0)
                bases = tc.cat((bases, bases1),0)
            if save_history and history1 is not None:
                if history is None:
                    history = history1
                else:
                    history = tc.cat((history, history), 0)
        return weight, bias, beta, margin, history, filters, bases, outsize_list

    def classify(self, input=None, prestr='', ismean=True):
        if input is None:
            input = self.input
        if input is None:
            return None
        assert isinstance(input, OrderedDataset)
        I = self.info(input)
        M = len(self.outsize_list)
        err = 0
#        print('M = {}, I.size = ({}, {})'.format(M, I.size(0), I.size(1)))
        for i, data in enumerate(I.cls_list):
            D = tc.Tensor()
#            ipdb.set_trace()
            for m in range(M):
                s, e = self.get_out_index(m)
#                print('i = {}, m = {}'.format(i, m))
                if ismean:
                    d = data[:, s:e]
                    if e-s > 1:
                        d = d.mean(1, keepdim=True)
                    else:
                        d = d.view(-1, 1)
                else:
                    d = tc.Tensor()
                    for n in range(M):
                        s, e = self.get_out_index(m, n)
                        if e-s > 1:
                            d0 = data[:, s:e].mean(1, keepdim=True)
                        else:
                            d0 = data[:, s:e].view(-1, 1)
                        d = tc.cat((d, d0), 1)
                    d, _ = d.max(1, keepdim=True)
                D = tc.cat((D, d), 1)
            ind = tc.argmax(D, dim=1)
            err += tc.sum(ind!=i)
#            print('i = {}, D.size = ({}, {}), err0 = {}'.format(i, D.size(0), D.size(1), err0))
        total = len(I)
        r = 100.0*(1.0 - float(err)/total)
        print(prestr + ': the accuracy is {:.4f}%, total samples are {}, '
              'misclassification number is {}'.format(r, total, err))
        return r
