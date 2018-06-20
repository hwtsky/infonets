# helper.py
"""
Created on Sat May  5 12:19:55 2018

@author: Wentao Huang
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from copy import deepcopy
import torch as tc
from ..param import Param


def is_number(data, is_raised=True, err="data", isnone=False):
    if data is None:
        if is_raised and (isnone is False):
            errors = err + " is None."
            raise ValueError(errors)
        else:
            return False
    else:
        import numbers
        if isinstance(data, numbers.Number):
            return True
        else:
            if is_raised:
                errors = err + " is not a Number type."
                raise TypeError(errors)
            else:
                return False


def is_numpy(data, is_raised=True, err="data", isnone=False):
    if data is None:
        if is_raised and (isnone is False):
            errors = err + " is None."
            raise ValueError(errors)
        else:
            return False
    else:
        import numpy as np
        if isinstance(data, np.ndarray): #(not type(data).__module__ == np.__name__): #
            return True
        else:
            if is_raised:
                errors = err + " is not a numpy.ndarray type."
                raise TypeError(errors)
            else:
                return False


def is_tensor(data, is_raised=True, err="data", isnone=False):
    if data is None:
        if is_raised and (isnone is False):
            errors = err + " is None."
            raise ValueError(errors)
        else:
            return False
    else:
        if isinstance(data, tc.Tensor):
            return True
        else:
            if is_raised:
                errors = err + " is not a torch.Tensor type."
                raise TypeError(errors)
            else:
                return False


def to_Tensor(data=None, dtype=None, requires_grad=False):#tc.float
    if data is None:
        data = tc.Tensor()
        if dtype is not None and dtype is not data.dtype:
            data = data.new_tensor(data.data, dtype=dtype,requires_grad=requires_grad) #as_tensor
        return data
    if isinstance(data, tc.Tensor):
        return data.new_tensor(data.data, dtype=dtype,requires_grad=requires_grad)
    else:
        import numpy as np
        import numbers
        ddtype = tc.get_default_dtype()
        if isinstance(data, np.ndarray):
            data = tc.from_numpy(data)
        elif isinstance(data, list):
            data = tc.tensor(data,requires_grad=requires_grad)
        elif isinstance(data, numbers.Number):
            data = tc.tensor([data],requires_grad=requires_grad)
        else:
            raise TypeError('to_Tensor: TypeError of data.')
        if dtype is None:
            if ddtype is not data.dtype:
                data = data.new_tensor(data, dtype=dtype,requires_grad=requires_grad)
        else:
            if dtype is not data.dtype:
                data = data.new_tensor(data.data, dtype=dtype,requires_grad=requires_grad)
        return data


def to_Param(data=None, requires_grad=True, dtype=None):
    if data is None:
        return None
    if isinstance(data, tc.Tensor):
        return Param(data, requires_grad, dtype)
    else:
        return Param(to_Tensor(data, dtype), requires_grad, dtype)


def copy_params(is_raised=False, err="copy_params", params=None, **kwargs):
    if (not bool(params)) or (not isinstance(params, dict)):
        p = OrderedDict()
    else:
        p = deepcopy(params)
    if bool(kwargs):
        p.update(kwargs)
    if is_raised and (not bool(p)):
        errors = err + ": parameter is empty."
        raise ValueError(errors)
    return p


def copy_params_ex(israise=False, err="copy_params_ex",
                   def_params=None, params=None, **kwargs):
    if (not bool(def_params)) or (not isinstance(def_params, dict)):
        p = OrderedDict()
    else:
        p = deepcopy(def_params)
    if bool(params):
        p.update(params)
    if bool(kwargs):
        p.update(kwargs)
    if israise and (not bool(p)):
        errors = err + ": params is empty."
        raise ValueError(errors)
    return p


def copy_params_in(israise=False, err="copy_params_in",
                   def_params=None, params=None, **kwargs):
    if (not bool(def_params)) or (not isinstance(def_params, dict)):
        p = OrderedDict()
    else:
        p = deepcopy(def_params)
    p0 = copy_params(is_raised=False, params=params, **kwargs)
    keys = set(p.keys())
    keys0 = set(p0.keys())
    keys1 = keys & keys0
    for key in keys1:
        p[key] = p0[key]
    if israise and (not bool(p)):
        errors = err + ": params is empty."
        raise ValueError(errors)
    return p


def params_to_Param(inplace=False, requires_grad=False,
                    dtype=None, params=None, **kwargs):
    if (not bool(params)) or (not isinstance(params, dict)):
        p = OrderedDict()
    else:
        if inplace:
            p = params
        else:
            p = deepcopy(params)
    if bool(kwargs):
        p.update(kwargs)
    if not bool(p):
        return p
    for key, val in p.items():
        if not isinstance(val, Param):
            val = to_Param(val, requires_grad, dtype)
            p[key] = val
    return p


def get_key_val(key=None, assign=None, is_raised=False,
                err="get_key_val", params=None, **kwargs):
    if bool(kwargs):
        params = copy_params(err=err, is_raised=False, params=params, **kwargs)
    n = 0
    val = None
    if isinstance(params, dict) and bool(params):
        if key in params.keys():
            val = params[key]
            n += 1
    if val is None and n == 0:
        if is_raised:
            raise ValueError(err+": No \'"+key+"\' key in dict")
        else:
            val = assign
    return val


def get_keys_vals(keyslist=None, assignlist=None, params=None, **kwargs):
    if bool(kwargs):
        params = copy_params(is_raised=False, params=params, **kwargs)
    if isinstance(keyslist, str):
        keyslist = list(keyslist)
    if (not isinstance(params, dict)) or (not bool(params)):
        vals = assignlist
    elif (not isinstance(keyslist, list)) or (len(keyslist)==0):
        vals = None
    else:
        L = len(keyslist)
        if not isinstance(assignlist, list):
            assignlist = list([assignlist])*L
        else:
            L1 = len(assignlist)
            assert L==L1, "get_keys_vals: len(keyslist) != len(assignlist)"
        vals = list([])
        pkeys = params.keys()#list()
        for key0, val0 in zip(keyslist, assignlist):
            if key0 in pkeys:
                val1 = params[key0]
            else:
                val1 = val0
            vals.append(val1)
    return vals


def raise_err(cond=True, err="raise_err", assign=False):
    if cond:
        raise ValueError(err)
    else:
        return assign


def flip(x, dim=0):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(tc.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]


def get_rootdir(up=0):
    import os, inspect
    up += 2
    parentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    if up <= 0:
        return parentdir
    for i in range(up):
        parentdir = os.path.dirname(parentdir)
    return parentdir


def path_join(path, *paths):
    from os.path import join
    return join(path, *paths)


def text_save(content, fname, fdir='results', mode='w'):#'a'
    # Try to save a list variable in txt file.
    fn = path_join(get_rootdir(), fdir, fname + '.txt')
    with open(fn, mode) as f:
        for i in content:
            f.write(i)