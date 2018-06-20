# fun.py
"""
Created on Sat May  5 20:56:15 2018

@author: Wentao Huang
"""
from __future__ import absolute_import, division
#import ipdb
#from copy import deepcopy
import torch as tc
from ..utils.data import OrderedDataset
from . import fun as obj_fun

LIST_ACTIVATION = ['plu', 'srelu', 'sigmoid', 'tanh', 'lptanh', 'lpsrelu']

def fun_exe(obj=None, fun=None, input=None, *args, **kwargs):
    if obj is None:
        F = getattr(obj_fun, fun, None)
    else:
        F = getattr(obj, fun, None)
    if F is None or input is None:
        return None
    if tc.is_tensor(input):
        return F(input, *args, **kwargs)
    elif isinstance(input, OrderedDataset):
        output = input.apply(obj_fun, fun, *args, **kwargs)
        return output
    else:
        return None


def relu(input, inplace=False):
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    if inplace:
        return tc.relu_(input)
    return tc.relu(input)


def lkrelu(input, gap=0.5, negative_slope=0.1, inplace=False):
    r"""
    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{lkrelu}(x) = \max(0, x) + \text{negative_slope} * \min(0, x+gap)`
    assert gap >= 0

    """
    if inplace:
        return plu(input.add_(0.5*gap), 0.5*gap, negative_slope, True)
    return plu(input.add(0.5*gap), 0.5*gap, negative_slope, True)
    

def plu(input, margin=0.5, negative_slope=1.0, inplace=False):
    r"""plu(input, margin=0.1, negative_slope=1.0, inplace=False) -> Tensor

    Applies the piecewise linear unit function element-wise,
    :math:`\text{plu}(x) = \max(0, x-margin) + \text{negative_slope} * \min(0, x+margin)`
    """
    if margin == 0.0:
        if inplace:
            input[input<0.0] = negative_slope*input[input<0.0] if negative_slope else 0.0
            return input
        else:
           input0 = input.clone()
           input0[input<0.0] = negative_slope*input[input<0.0] if negative_slope else 0.0
           return input0
    elif margin > 0.0:
        if inplace:
            input[tc.abs(input)<=margin] = 0.0
            input[input>margin] = input[input>margin] - margin
            input[input<-margin] = negative_slope*(input[input<-margin] + margin) if negative_slope else 0.0
            return input
        else:
           input0 = input.clone()
           input0[tc.abs(input)<=margin] = 0.0
           input0[input>margin] = input0[input>margin] - margin
           input0[input<-margin] = negative_slope*(input[input<-margin] + margin) if negative_slope else 0.0
           return input0
    else:
        if inplace:
            input[tc.abs(input)<=-margin] = (1+negative_slope)*input[tc.abs(input)<=-margin] - (1-negative_slope)*margin
            input[input>-margin] = input[input>-margin] - margin
            input[input<margin] = negative_slope*(input[input<margin] + margin) if negative_slope else 0.0
            return input
        else:
           input0 = input.clone()
           input0[tc.abs(input)<=-margin] = (1+negative_slope)*input[tc.abs(input)<=-margin] - (1-negative_slope)*margin
           input0[input>-margin] = input0[input>-margin] - margin
           input0[input<margin] = negative_slope*(input[input<margin] + margin) if negative_slope else 0.0
           return input0

def plu_grad(input, margin=0.5, negative_slope=1.0):
    r"""plu_grad(input, margin=0.1, negative_slope=1.0) -> Tensor

    Applies the gradient of the piecewise linear unit function element-wise,
    """
    if margin == 0.0:
        output = tc.ones(input.size())
        output[input<0.0] = negative_slope
        return output
    elif margin > 0.0:
        output = tc.ones(input.size())
        output[tc.abs(input)<=margin] = 0.0
        output[input<-margin] = negative_slope
        return output
    else:
        output = tc.ones(input.size())
        output[tc.abs(input)<=-margin] = 1.0 + negative_slope
        output[input<margin] = negative_slope
        return output

def srelu(input, beta=1.0, inplace=False):
    r"""srelu(input, beta=1.0, inplace=False) -> Tensor
    
    Applies the soft rectified linear unit function element-wise.
    :math:`\text{srelu}(x) = 1/beta*log(1+exp(beta*input))`
    
    """
    if inplace:
        if beta == 1.0:
            return input.exp_().add_(1.0).log_()
        return input.mul_(beta).exp_().add_(1.0).log_().mul_(1.0/beta)
    else:
        if beta == 1.0:
            return input.exp().add_(1.0).log_()
        return input.mul(beta).exp_().add_(1.0).log_().mul_(1.0/beta)

def srelu_grad(input, beta=1.0):
    r"""srelu_grad(input, beta=1.0) -> Tensor

    Applies the gradient of the soft rectified linear unit function element-wise.
    """
    return input.mul(beta).sigmoid_()

def sigmoid(input, beta=1.0, inplace=False):
    r"""sigmoid(input, beta=1.0, inplace=False) -> Tensor

    Applies the sigmoid function element-wise.
    """
    if inplace:
        input.mul_(beta).sigmoid_()
    return input.mul(beta).sigmoid_()


def sigmoid_grad(input, beta=1.0):
    r"""sigmoid_grad(input, beta=1.0) -> Tensor

    Applies the gradient of the sigmoid function element-wise.
    """
    if beta == 1.0:
        grad = input.sigmoid()
        return grad*(1.0 - grad)
    grad = input.mul(beta).sigmoid_()
    return beta*grad*(1.0 - grad)

def lpsigmoid(input, alpha=0.5, beta=1.0, inplace=False):
    r"""lpsigmoid(input, alpha=0.5, beta=1.0, inplace=False) -> Tensor

    Applies the gradient of the linear plus sigmoid function element-wise.
    :math:`\text{lpsigmoid}(x) = alpha*sigmoid(beta*input) + (1-alpha)*x`
    
    """
    if inplace:
        x = input.mul(1.0-alpha)
        if beta == 1.0:
            return input.add_(input.sigmoid().mul_(alpha)).add_(x)
        return input.mul(beta).sigmoid_().mul_(alpha).add_(x)
    else:
        if beta == 1.0:
            return input.sigmoid().mul_(alpha).add_(input.mul(1.0-alpha))
        return input.mul(beta).sigmoid_().mul_(alpha).add_(input.mul(1.0-alpha))

def lpsigmoid_grad(input, alpha=0.5, beta=1.0):
    r"""lpsigmoid_grad(input, alpha=0.5, beta=1.0) -> Tensor

    Applies the gradient of the linear plus sigmoid function element-wise.
    """
    if beta == 1.0:
        x = input.sigmoid()
    else:
        x = input.mul(beta).sigmoid_()
    return x.mul_(1.0-x).mul_(alpha*beta).add_(1.0-alpha)

def tanh(input, beta=1.0, inplace=False):
    r"""tanh(input, beta=1.0, inplace=False) -> Tensor

    Applies the tanh function element-wise.
    """
    if inplace:
        input.mul_(beta).tanh_()
    return input.mul(beta).tanh_()

def tanh_grad(input, beta=1.0):
    r"""tanh_grad(input, beta=1.0) -> Tensor

    Applies the gradient of the tanh function element-wise.
    """
    if beta == 1.0:
        return 1.0 - input.tanh().pow_(2)
    return beta*(1.0 - input.tanh().pow_(2))

def lpstanh(input, beta=1.0, sign=1.0, inplace=False):
    r"""lpstanh(input, beta=1.0, sign=1.0, inplace=False) -> Tensor

    Applies the gradient of the linear plus tanh function element-wise.
    :math:`\text{lpstanh}(x) = sign*tanh(beta*input) + beta*x`
    
    """
    if inplace:
        if beta == 1.0:
            return input.add_(input.tanh().mul_(1.0))
        input.mul_(beta)
        return input.add_(input.tanh().mul_(1.0))
    else:
        if beta == 1.0:
            return input.add(input.tanh().mul_(1.0))
        input = input.mul(beta)
        return input.add_(input.tanh().mul_(1.0))

def lpstanh_grad(input, beta=1.0, sign=1.0):
    r"""lpstanh_grad(input, beta=1.0, sign=1.0) -> Tensor

    Applies the gradient of the linear plus tanh function element-wise.
    """
    if beta == 1.0:
        return input.tanh().pow_(2).mul_(-sign).add_(1.0+sign)
    return input.mul(beta).tanh_().pow_(2).mul_(-sign*beta).add_(beta+beta*sign)

def lptanh(input, alpha=0.5, beta=1.0, inplace=False):
    r"""lptanh(input, alpha=0.5, beta=1.0, inplace=False) -> Tensor

    Applies the gradient of the linear plus tanh function element-wise.
    :math:`\text{lptanh}(x) = alpha*tanh(beta*input) + (1-alpha)*x`
    
    """
    if inplace:
        x = input.mul(1.0-alpha)
        if beta == 1.0:
            return input.tanh_().mul_(alpha).add_(x)
        return input.mul_(beta).tanh_().mul_(alpha).add_(x)
    else:
        if beta == 1.0:
            return input.tanh().mul_(alpha).add_(input.mul(1.0-alpha))
        return input.mul(beta).tanh_().mul_(alpha).add_(input.mul(1.0-alpha))

def lptanh_grad(input, alpha=0.5, beta=1.0):
    r"""lptanh_grad(input, alpha=0.5, beta=1.0) -> Tensor

    Applies the gradient of the linear plus tanh function element-wise.
    """
    if beta == 1.0:
        return input.tanh().pow_(2).mul_(-alpha).add_(1.0)
    return input.tanh().pow_(2).mul_(-alpha*beta).add_(alpha*(beta-1.0)+1.0)

def lpsrelu(input, alpha=0.5, beta=1.0, inplace=False):
    r"""lpsrelu(input, alpha=0.5, beta=1.0, inplace=False) -> Tensor

    Applies the linear plus soft rectified linear unit function element-wise.
    :math:`\text{lpsrelu}(x) = alpha/beta*log(1+exp(beta*input)) + (1-alpha)*x`
    
    """
    if inplace:
        x = input.mul(1.0-alpha)
        if beta == 1.0:
            return input.exp_().add_(1.0).log_().mul_(alpha).add_(x)
        return input.mul_(beta).exp_().add_(1.0).log_().mul_(alpha/beta).add_(x)
    else:
        if beta == 1.0:
            return input.exp().add_(1.0).log_().mul_(alpha).add_(input.mul(1.0-alpha))
        return input.mul(beta).exp_().add_(1.0).log_().mul_(alpha/beta).add_(input.mul(1.0-alpha))

def lpsrelu_grad(input, alpha=0.5, beta=1.0):
    r"""lpsrelu_grad(input, alpha=0.5, beta=1.0) -> Tensor

    Applies the gradient of the linear plus soft rectified linear unit function element-wise.
    """
    if beta == 1.0:
        return input.sigmoid().mul_(alpha).add_(1.0-alpha)
    return input.mul(beta).sigmoid_().mul_(alpha).add_(1.0-alpha)
