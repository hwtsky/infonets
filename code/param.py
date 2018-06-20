# param.py
"""
Created on Sat May  5 10:47:19 2018

@author: Wentao Huang
"""
import torch as tc


class Param(tc.nn.Parameter):
    r"""A kind of Tensor that is to be considered a module parameter.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `False`
    """
    def __new__(cls, data=None, requires_grad=True, dtype=None):
        if data is None:
            data = tc.tensor(tc.Tensor(), dtype=dtype)
        elif isinstance(data, tc.Tensor):
            data = tc.tensor(data.detach(), dtype=dtype)
        else:
            raise TypeError('Param: data must be a Tensor type.')
        return tc.Tensor._make_subclass(cls, data, requires_grad)

    def __repr__(self):
        return 'Param containing:\n' + super(tc.nn.Parameter, self).__repr__()
