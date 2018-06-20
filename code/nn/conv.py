# conv.py
"""
Created on Sun Jun 10 22:43:32 2018

@author: Wentao Huang
"""
#import ipdb
import math
from copy import deepcopy
from collections import OrderedDict, Iterable
import torch as tc
import torch.nn.functional as F
from .linear import Linear
from .. import utils
from ..utils.data import OrderedDataset
from ..utils.helper import to_Param, copy_params_in, get_keys_vals
from ..utils.methods import get_outsize
from .. import nn

PARAMS_CONV = OrderedDict()
PARAMS_CONV['learning_method'] = 'lpsrelu_const'
PARAMS_CONV['conn_method'] = 'Conn' # 'Conn', 'Conn1', 'Conn2'

PARAMS_CONV['extraction_step'] = 1
PARAMS_CONV['indent'] = 0
PARAMS_CONV['max_patches'] = None # int(1e+5) #
PARAMS_CONV['max_iter_samples'] = 0


def _ntuple(n):
    from itertools import repeat
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)

class _ConvNd(Linear):

    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, padding, dilation, 
                 transposed, output_padding, groups, 
                 bias, inplace, dtype=None, name='ConvNd'):
        super(_ConvNd, self).__init__(bias=bias, dtype=dtype, name=name)
        if in_channels is not None and in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels is not None and out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.inplace = inplace
        self.conn = None
        self.dim = 0
        self.set_weight(dtype=dtype)
        self.register_buffer('bases', None)
        self.register_buffer('filters', None)
        self.register_buffer('history', None)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def set_weight(self, weight=None, requires_grad=True, dtype=None):
        if not hasattr(self, 'kernel_size'):
            return super(_ConvNd, self).set_weight(weight, requires_grad, dtype)
        if weight is None:
            if self.in_channels is not None and self.out_channels is not None:
                if self.transposed:
                    weight = to_Param(tc.Tensor(
                        self.in_channels, self.out_channels//self.groups, *self.kernel_size))# 
                else:
                    weight = to_Param(tc.Tensor(
                        self.out_channels, self.in_channels//self.groups, *self.kernel_size))# 
                n = self.in_channels
                for k in self.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                weight.data.uniform_(-stdv, stdv)
                weight = to_Param(weight.t(), requires_grad, dtype)
                self.out_features = self.out_channels
                self.in_features = n
            else:
                self.out_features = self.out_channels
                if self.in_channels is not None:
                    n = self.in_channels
                    for k in self.kernel_size:
                        n *= k
                    self.in_features = n
                else:
                    self.in_features = None
        else:
#            ipdb.set_trace()
            weight = to_Param(weight, requires_grad, dtype)
            s = weight.size()
            ls = len(s)
            if ls == 0 or s[0] == 0:
                raise ValueError('The length of shape of weight cannot be zero!')
            elif ls==1:
                weight = weight.veiw(1,s[0])
            s = weight.size()
            m = 1
            for i in s[1:]:
                m *= i
            self.in_features = m
            if self.transposed:
                weight.transpose_(0, 1)
                self.out_channels = self.out_features = s[0]*self.groups#
                self.in_channels = s[1]
            else:
                self.out_channels = self.out_features = s[0]
                self.in_channels = s[1]*self.groups
        if self.in_channels is not None and self.in_channels % self.groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if self.out_channels is not None and self.out_channels % self.groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.register_parameter('weight', weight)
        return weight

    def set_in_features(self, in_channels=None, kernel_size=None):
        in_channels = self.in_channels if in_channels is None else in_channels
        kernel_size = self.kernel_size if kernel_size is None else kernel_size
        if isinstance(kernel_size, Iterable):
            k = 1
            for i in kernel_size:
                k *= i
        else:
            k = kernel_size
        self.in_features = int(in_channels*k)
        return self.in_features

    def get_out_channels(self, in_channels=None, kernel_size=None, 
                         groups=None, conn=None):
        if self.out_channels is not None:
            return self.out_channels
        if in_channels is None:
            in_channels = self.in_channels
        if kernel_size is None:
            kernel_size = self.kernel_size
        if groups is None:
            groups = self.groups
        if conn is None:
            conn = self.conn
        assert conn is not None
        refsize = conn.get_layer_param(0, key='refsize')
        out_zoom = conn.get_layer_param(1, key='out_zoom')
        outsize = conn.get_layer_param(1, key='outsize')
        outscale = conn.get_layer_param(1, key='outscale')
        if refsize is None:
            refsize = self.set_in_features(in_channels/groups, kernel_size)
        return groups * get_outsize(refsize, outsize, outscale, out_zoom)

    def get_history(self):
        return self.history

    def extra_repr(self):
        s = ('in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        s = s.format(**self.__dict__)
        if tc.is_tensor(self.bias):
            s += ', bias={:.2e}'.format(self.bias.mean().item())
        s += ', inplace=True' if self.inplace else ', inplace=False'
        s += ', name={}'.format(self.name)
        return s

    def forward(self, input=None, inplace=None):
        if input is None:
            input = self.input
        if input is None:
            return None
        if inplace is None:
            inplace = self.inplace
        else:
            self.inplace = inplace
        weight, bias = self.get_weight_bias(self.weight, self.bias, self.beta, 
                                            self.margin, self.coff_a)
        if self.dim == 1:
            fun = 'conv1d'
        elif self.dim == 2:
            fun = 'conv2d'
        elif self.dim == 3:
            fun = 'conv3d'
        else:
            fun = None
        if isinstance(input, tc.Tensor):
            f = getattr(F, fun, None)
            assert f is not None
            if inplace:
                input.data = f(input, weight, bias, self.stride,
                               self.padding, self.dilation, self.groups).data
                return input
            else:
                return f(input, weight, bias, self.stride,
                         self.padding, self.dilation, self.groups)
        elif isinstance(input, OrderedDataset):
            if inplace:
                return input.apply_(F, fun, weight, bias, self.stride,
                               self.padding, self.dilation, self.groups)
            else:
                return input.apply(F, fun, weight, bias, self.stride,
                               self.padding, self.dilation, self.groups)
        else:
            return None

    def slice(self, input=None, groups=None, ith_gp=0):
        if input is None:
            input = self.input
        if groups is None:
            groups = self.groups
        if groups <= 1:
            return input
        L = input.size(1)
        m = int(L/groups)
        assert L%groups == 0, '{}.slices(input, groups): input.size(1) must be divisible by groups'.format(self.name)
        cat_end = input.cat_end
        max_iter_samples = input.max_iter_samples
        input.set_sample_iter(0)
        K = range(0, L, m)
        output = []
        for X, _ in input:
            output.append(X[:, K[ith_gp]:(K[ith_gp]+m)])
        input.set_sample_iter(max_iter_samples, cat_end)
        output = OrderedDataset(output, max_iter_samples=max_iter_samples, cat_end=cat_end)
        output.cls_labels = deepcopy(input.cls_labels)
        return output
    
    def slices(self, input=None, groups=None, inplace=True):
        if input is None:
            input = self.input
        if groups is None:
            groups = self.groups
        if groups <= 1:
            return input
        L = input.size(1)
        m = int(L/groups)
        assert L%groups == 0, '{}.slices(input, groups): input.size(1) must be divisible by groups'.format(self.name)
        cat_end = input.cat_end
        max_iter_samples = input.max_iter_samples
        input.set_sample_iter(0)
        output = []
        k = 0
        for X, _ in input:
            Y = None
#            ipdb.set_trace()
            for i in range(0, L, m):
                x = X[:,i:(i+m)]
                Y = x if Y is None else tc.cat((Y, x), 0)
            X = x = None
            if inplace:
                input.cls_list[k] = Y
            else:
                output.append(Y)
            k += 1
        input.set_sample_iter(max_iter_samples, cat_end)
        if inplace:
            return input
        else:
            output = OrderedDataset(output, max_iter_samples=max_iter_samples, cat_end=cat_end)
            output.cls_labels = deepcopy(input.cls_labels)
            return output

    def train_exe(self, input=None, params=None, **kwargs):
        if self.conn is not None and self.learning_method is not None:
            input = self.input if input is None else input
            if input is None:
                raise ValueError("{}.train_exe(input): input is None".format(self.name))
            else:
                assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
            p = self._parameters_train
            p = self._parameters_train = copy_params_in(
                                         def_params=p, params=params, **kwargs)
            assignlist = [1, 0, None, 0]
            keyslist = ['extraction_step', 'indent', 'max_patches', 'max_iter_samples']
            [extraction_step, indent, max_patches,
             max_iter_samples] = get_keys_vals(keyslist, assignlist, params=p)
            kernel_size = self.kernel_size
            seed = self.conn.get_layer_param(1, 'seed')
            if self.dim == 1:
                OD = 'Ordered1DPatches'
            elif self.dim == 2:
                OD = 'Ordered2DPatches'
            elif self.dim == 3:
                OD = 'Ordered3DPatches'
            else:
                OD = None
            bias_requires_grad = self.conn.get_layer_param(1,'bias_requires_grad')
            save_bases_filters = self.conn.get_parameters_train('save_bases_filters')
            save_inter_vals = self.conn.get_parameters_train('save_inter_vals')
            save_history = self.conn.get_parameters_train('save_history')
#            ipdb.set_trace()
            f = getattr(utils, OD, None)
            assert f is not None
            input = f(input, kernel_size, extraction_step,
                      indent, max_patches, seed, None, None, max_iter_samples)
            self.in_channels = input.size(1)
            groups = self.groups
            self.set_in_features()
            n = int(self.in_channels/groups)
            if groups == 1:
                input.view_(-1)
                weight, bias, beta, margin, history = self.conn.train_exe(
                                                input, params=params, **kwargs)
                m = weight.size(0)
                shape = (m, n) + kernel_size
                weight = weight.view(shape)
                if save_bases_filters:
                    bases = self.conn.bases.view(shape)
                    filters = self.conn.filters.view(shape)
            else:
                weight = None
                bias = None
                history = None
                bases = None
                filters = None
                for i in range(groups):
                    X = self.slice(input, groups, i).view_(-1)
                    weight0, bias0, beta, margin, history0 = self.conn.train_exe(
                                                    X, params=params, **kwargs)
                    m = weight0.size(0)
                    shape = (m, n) + kernel_size
                    weight0 = weight0.view(shape)
#                    ipdb.set_trace()
                    if bias0 is not None and len(bias0)==1:
                        bias0 = tc.ones(m)*bias0
                    weight = weight0 if weight is None else tc.cat((weight, weight0), 0)
                    if bias0 is not None:
                        bias = bias0 if bias is None else tc.cat((bias, bias0))
                    if save_history:
                        history = history0 if history is None else tc.cat((history, history0))
                    if save_bases_filters:
                        bases0 = self.conn.bases.view(shape)
                        filters0 = self.conn.filters.view(shape)
                        bases = bases0 if bases is None else tc.cat((bases, bases0),0)
                        filters = filters0 if filters is None else tc.cat((filters, filters0),0)
            self.out_channels = self.out_features = weight.size(0)
            if bias is not None and len(bias)==1:
                bias = tc.ones(self.out_channels)*bias
            weight = self.set_weight(weight, True)
            bias = self.set_bias(bias, bias_requires_grad)
            beta = self.set_beta(beta)
            margin = self.set_margin(margin)
            if save_history:
                self.history = history
            self.conn.history = self.conn.layers[1].history = None
            if save_bases_filters:
                self.bases = bases
                self.filters = filters
            self.conn.bases = self.conn.filters = None
            self.conn.weight = self.conn.bias = None
            if not save_inter_vals:
                self.conn.empty_buffers()
                self.conn.layers[0].empty_buffers()
                self.conn.layers[1].empty_buffers()
                self.conn.layers[0].empty_parameters()
                self.conn.layers[1].weight = None
                self.conn.layers[1].bias = None
            return weight, bias, beta, margin, history
        else:
            return None


class Conv1d(_ConvNd):

    def __init__(self, in_channels=None, out_channels=None,
                 kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=None,
                 inplace=True, input=None, dtype=None, 
                 name='Conv1d', params=None, **kwargs):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        if input is not None:
            in_channels = input.size(1)
        p = PARAMS_CONV.copy()
        p = copy_params_in(def_params=p, params=params, **kwargs)
        keyslist = ['learning_method', 'conn_method']
        assignlist = ['lpsrelu_const', 'Conn']
        [learning_method, conn_method] = get_keys_vals(keyslist, assignlist, params=p)
        conn = getattr(nn, conn_method, None)
        if conn is None:
            raise ValueError('{}.__init__: No such conn_method, {}'.format(
                              name, conn_method))
        conn = conn(bias=bias, beta=None, margin=None,
                         learning_method=learning_method,
                         dtype=dtype, params=params, **kwargs)
        self.in_channels = in_channels
        self.out_channels = None
        if out_channels is None:
            out_channels = self.get_out_channels(in_channels, kernel_size, groups, conn)
        assert out_channels is not None
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
               padding, dilation, False, _single(0), groups, bias, inplace, dtype, name)
        self.input = input
        self.conn = conn
        self._parameters_train = p
        self._learning_method = learning_method
        self.inplace = inplace
        self.dim = 1
        self.register_parameter('bias', conn.bias)
        self.register_parameter('margin', conn.margin)
        self.register_parameter('beta', conn.beta)
        self.register_parameter('coff_a', conn.coff_a)

    def get_outsize(self, insize=None, kernel_size=None,
                    stride=None, padding=None, dilation=None):
        if insize is None and self.input is not None:
            insize = self.input.size(-1)
        assert insize is not None
        kernel_size = kernel_size or self.kernel_size
        stride = stride or self.stride
        padding = padding or self.padding
        dilation = dilation or self.dilation
        insize = _single(insize)
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        H = int((insize[0]+2.0*padding[0]-dilation[0]*(kernel_size[0]-1.0)-1.0)/stride[0]+1.0)
        L = 0 if self.input is None else self.input.size(0)
        C = 0 if self.out_channels is None else self.out_channels
        return (L, C, H)


class Conv2d(_ConvNd):

    def __init__(self, in_channels=None, out_channels=None,
                 kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=None,
                 inplace=True, input=None, dtype=None, 
                 name='Conv2d', params=None, **kwargs):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if input is not None:
            in_channels = input.size(1)
        p = PARAMS_CONV.copy()
        p = copy_params_in(def_params=p, params=params, **kwargs)
        keyslist = ['learning_method', 'conn_method']
        assignlist = ['lpsrelu_const', 'Conn']
        [learning_method, conn_method] = get_keys_vals(keyslist, assignlist, params=p)
        conn = getattr(nn, conn_method, None)
        if conn is None:
            raise ValueError('{}.__init__: No such conn_method, {}'.format(
                              name, conn_method))
        conn = conn(bias=bias, beta=None, margin=None,
                         learning_method=learning_method,
                         dtype=dtype, params=params, **kwargs)
        self.in_channels = in_channels
        self.out_channels = None
        if out_channels is None:
            out_channels = self.get_out_channels(in_channels, kernel_size, groups, conn)
        assert out_channels is not None
#        ipdb.set_trace()
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
               padding, dilation, False, _pair(0), groups, bias, inplace, dtype, name)
        self.input = input
        self.conn = conn
        self._parameters_train = p
        self._learning_method = learning_method
        self.inplace = inplace
        self.dim = 2
        self.register_parameter('bias', conn.bias)
        self.register_parameter('margin', conn.margin)
        self.register_parameter('beta', conn.beta)
        self.register_parameter('coff_a', conn.coff_a)

    def get_outsize(self, insize=None, kernel_size=None,
                    stride=None, padding=None, dilation=None):
        if insize is None and self.input is not None:
            insize = (self.input.size(-2), self.input.size(-1))
        assert insize is not None
        kernel_size = kernel_size or self.kernel_size
        stride = stride or self.stride
        padding = padding or self.padding
        dilation = dilation or self.dilation
        insize = _pair(insize)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        H = int((insize[0]+2.0*padding[0]-dilation[0]*(kernel_size[0]-1.0)-1.0)/stride[0]+1.0)
        W = int((insize[1]+2.0*padding[1]-dilation[1]*(kernel_size[1]-1.0)-1.0)/stride[1]+1.0)
        L = 0 if self.input is None else self.input.size(0)
        C = 0 if self.out_channels is None else self.out_channels
        return (L, C, H, W)


class Conv3d(_ConvNd):

    def __init__(self, in_channels=None, out_channels=None,
                 kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=None,
                 inplace=True, input=None, dtype=None, 
                 name='Conv3d', params=None, **kwargs):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        if input is not None:
            in_channels = input.size(1)
        p = PARAMS_CONV.copy()
        p = copy_params_in(def_params=p, params=params, **kwargs)
        keyslist = ['learning_method', 'conn_method']
        assignlist = ['lpsrelu_const', 'Conn']
        [learning_method, conn_method] = get_keys_vals(keyslist, assignlist, params=p)
        conn = getattr(nn, conn_method, None)
        if conn is None:
            raise ValueError('{}.__init__: No such conn_method, {}'.format(
                              name, conn_method))
        conn = conn(bias=bias, beta=None, margin=None,
                         learning_method=learning_method,
                         dtype=dtype, params=params, **kwargs)
        self.in_channels = in_channels
        self.out_channels = None
        if out_channels is None:
            out_channels = self.get_out_channels(in_channels, kernel_size, groups, conn)
        assert out_channels is not None
#        ipdb.set_trace()
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
               padding, dilation, False, _triple(0), groups, bias, inplace, dtype, name)
        self.input = input
        self.conn = conn
        self._parameters_train = p
        self._learning_method = learning_method
        self.inplace = inplace
        self.dim = 3
        self.register_parameter('bias', conn.bias)
        self.register_parameter('margin', conn.margin)
        self.register_parameter('beta', conn.beta)
        self.register_parameter('coff_a', conn.coff_a)

    def get_outsize(self, insize=None, kernel_size=None,
                    stride=None, padding=None, dilation=None):
        if insize is None and self.input is not None:
            insize = (self.input.size(-3), self.input.size(-2), self.input.size(-1))
        assert insize is not None
        kernel_size = kernel_size or self.kernel_size
        stride = stride or self.stride
        padding = padding or self.padding
        dilation = dilation or self.dilation
        insize = _triple(insize)
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        D = int((insize[0]+2.0*padding[0]-dilation[0]*(kernel_size[0]-1.0)-1.0)/stride[0]+1.0)
        H = int((insize[1]+2.0*padding[1]-dilation[1]*(kernel_size[1]-1.0)-1.0)/stride[1]+1.0)
        W = int((insize[2]+2.0*padding[2]-dilation[2]*(kernel_size[2]-1.0)-1.0)/stride[2]+1.0)
        L = 0 if self.input is None else self.input.size(0)
        C = 0 if self.out_channels is None else self.out_channels
        return (L, C, D, H, W)
