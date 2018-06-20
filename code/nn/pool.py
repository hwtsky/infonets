# pool.py
"""
Created on Wed Jun 13 23:26:19 2018

@author: Wentao Huang
"""
#import ipdb
from collections import Iterable
import torch as tc
import torch.nn.functional as F
from .base import Base
from ..utils.data import OrderedDataset


def _ntuple(n):
    from itertools import repeat
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)
_triple = _ntuple(3)


class MaxPool1d(Base):

    def __init__(self, kernel_size=2, stride=None, padding=0, 
                 dilation=1, return_indices=False, ceil_mode=False, 
                 inplace=True, input=None, name='MaxPool1d'):
        super(MaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.input = input
        self.inplace = inplace
        self.name = name
        self.dim = 1

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)
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
        if self.dim == 1:
            fun = 'max_pool1d'
        elif self.dim == 2:
            fun = 'max_pool2d'
        elif self.dim == 3:
            fun = 'max_pool3d'
        else:
            fun = None
        if isinstance(input, tc.Tensor):
            f = getattr(F, fun, None)
            assert f is not None
            if inplace:
                input.data = f(input, self.kernel_size, self.stride, 
                               self.padding, self.dilation, self.ceil_mode, 
                               self.return_indices).data
                return input
            else:
                return f(input, self.kernel_size, self.stride, self.padding, 
                         self.dilation, self.ceil_mode, self.return_indices)
        elif isinstance(input, OrderedDataset):
            if inplace:
                return input.apply_(F, fun, self.kernel_size, self.stride, 
                                   self.padding, self.dilation, 
                                   self.ceil_mode, self.return_indices)
            else:
                return input.apply(F, fun, self.kernel_size, self.stride, 
                                   self.padding, self.dilation, 
                                   self.ceil_mode, self.return_indices)
        else:
            return None
    
    
class MaxPool2d(MaxPool1d):
    
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, 
                 inplace=True, input=None, name='MaxPool2d'):
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                        return_indices, ceil_mode, inplace, input, name)
        self.kernel_size = kernel_size# _pair(kernel_size)
        stride = stride or kernel_size
        self.stride = stride#_pair(stride)
        self.dim = 2
        
        
class MaxPool3d(MaxPool1d):
    
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, 
                 inplace=True, input=None, name='MaxPool3d'):
        super(MaxPool3d, self).__init__(kernel_size, stride, padding, dilation,
                                        return_indices, ceil_mode, inplace, input, name)
        self.kernel_size = kernel_size#_triple(kernel_size)
        stride = stride or kernel_size
        self.stride = stride#_triple(stride)
        self.dim = 3
    
    