# act.py
"""
Created on Sat May  5 22:24:18 2018

@author: Wentao Huang
"""
#import torch as tc
#from torch.nn import functional as F
#from ..utils.helper import is_number, to_Param
from .base import Base
from .fun import *


class PLU(Base):
    r"""
    Applies the piecewise linear unit function element-wise,
    :math:`\text{PLU}(x) = \max(0, x-margin) + \text{negative_slope} * \min(0, x+margin)`
    """
    def __init__(self, margin=0.5, negative_slope=1.0, inplace=False, name='PLU'):
        super(PLU, self).__init__()
        self.margin = margin
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.set_name(name)
        self.training = False

    def grad(self, input):
        return plu_grad(input, self.margin, self.negative_slope)
    
    def forward(self, input):
        return plu(input, self.margin, self.negative_slope, self.inplace)

    def extra_repr(self):
        return 'margin={}, negative_slope={}, inplace={}, name={}'.format(
                self.margin, self.negative_slope, self.inplace, self._name)


class SReLU(Base):
    r"""
    Applies the soft rectified linear unit function element-wise.
    """
    def __init__(self, beta=1.0, inplace=False, name='SReLU'):
        super(SReLU, self).__init__()
        self.beta = beta
        self.inplace = inplace
        self.set_name(name)
        self.training = False

    def grad(self, input):
        return srelu_grad(input, self.beta)
    
    def forward(self, input):
        return srelu(input, self.beta, self.inplace)

    def extra_repr(self):
        return 'beta={}, inplace={}, name={}'.format(
                self.beta, self.inplace, self._name)


class Sigmoid(Base):
    r"""
    Applies the sigmoid function element-wise.
    """
    def __init__(self, beta=1.0, inplace=False, name='Sigmoid'):
        super(Sigmoid, self).__init__()
        self.beta = beta
        self.inplace = inplace
        self.set_name(name)
        self.training = False

    def grad(self, input):
        return sigmoid_grad(input, self.beta)
    
    def forward(self, input):
        if self.inplace:
            if self.beta == 1.0:
                return input.sigmoid_()
            return input.mul_(self.beta).sigmoid_()
        else:
            if self.beta == 1.0:
                return input.sigmoid()
            return input.mul(self.beta).sigmoid_()

    def extra_repr(self):
        return 'beta={}, inplace={}, name={}'.format(
                self.beta, self.inplace, self._name)


class Tanh(Base):
    r"""
    Applies the tanh function element-wise.
    """
    def __init__(self, beta=1.0, inplace=False, name='Tanh'):
        super(Tanh, self).__init__()
        self.beta = beta
        self.inplace = inplace
        self.set_name(name)
        self.training = False

    def grad(self, input):
        return tanh_grad(input, self.beta)
    
    def forward(self, input):
        if self.inplace:
            if self.beta == 1.0:
                return input.tanh_()
            return input.mul_(self.beta).tanh_()
        else:
            if self.beta == 1.0:
                return input.tanh()
            return input.mul(self.beta).tanh_()

    def extra_repr(self):
        return 'beta={}, inplace={}, name={}'.format(
                self.beta, self.inplace, self._name)

class LPTanh(Base):
    r"""
    Applies the linear plus tanh function element-wise.
    """
    def __init__(self, beta=1.0, inplace=False, name='Tanh'):
        super(LPTanh, self).__init__()
        self.beta = beta
        self.inplace = inplace
        self.set_name(name)
        self.training = False

    def grad(self, input):
        return lptanh_grad(input, self.beta)
    
    def forward(self, input):
        return lptanh(input, self.beta, self.inplace)

    def extra_repr(self):
        return 'beta={}, inplace={}, name={}'.format(
                self.beta, self.inplace, self._name)
        
class LPSReLU(Base):
    r"""
    Applies the linear plus soft rectified linear unit function element-wise.
    """
    def __init__(self, alpha=0.5, beta=1.0, inplace=False, name='LPSReLU'):
        super(LPSReLU, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.inplace = inplace
        self.set_name(name)
        self.training = False

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0.0:
           raise ValueError("{}: The parameter alpha must be greater than zero".format(self._name))
        elif value > 1.0:
           raise ValueError("{}: The parameter alpha must be less than one".format(self._name))    
        else:
            self._alpha = value
            
    def grad(self, input):
        return lpsrelu_grad(input, self.alpha, self.beta)
    
    def forward(self, input):
        return lpsrelu(input, self.alpha, self.beta, self.inplace)

    def extra_repr(self):
        return 'alpha={}, beta={}, inplace={}, name={}'.format(
                self.alpha, self.beta, self.inplace, self._name)
        
        