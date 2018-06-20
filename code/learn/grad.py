# grad.py
"""
Created on Fri May 25 19:10:00 2018

@author: Wentao Huang
"""
from torch.autograd import Function


class Grad(Function):
    r"""Records operation history and defines formulas for differentiating ops.

    Each function object is meant to be used only once (in the forward pass).

    Attributes:
        requires_grad: Boolean indicating whether the :func:`backward` will
            ever need to be called.
    """
    
    @staticmethod
    def forward(ctx, input, C, bias=None, beta=1.0, 
                isorth=True, eps=1e-6, *args, **kwargs):
        r"""Performs the operation.

        This function is to be overridden by all subclasses.

        It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).

        The context can be used to store tensors that can be then retrieved
        during the backward pass.
        """
        return NotImplementedError

    @staticmethod
    def backward(ctx, grad_output=None):
        r"""Defines a formula for differentiating the operation.

        It must accept a context ctx as the first argument, followed by as many
        outputs did :func:`forward` return, and it should return as many
        tensors, as there were inputs to :func:`forward`. Each argument is the
        gradient w.r.t the given output, and each returned value should be the
        gradient w.r.t. the corresponding input.

        The context can be used to retrieve tensors saved during the forward
        pass.
        """
#        d_input=d_C=d_b=d_beta=d_isorth=d_eps=None
        dC, db, argnum = ctx.saved_variables
        output = [None]*int(argnum)
        if ctx.needs_input_grad[1]:
            output[1] = dC #d_C = dC
        if ctx.needs_input_grad[2]:
            output[2] = db #d_b = db
        if grad_output is not None:
            output[1] = grad_output * output[1]
            if output[2] is not None:
                output[2] = grad_output * output[2]
        return tuple(output) # d_input, d_C, d_b, d_beta, d_isorth, d_eps

