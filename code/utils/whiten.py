# whiten.py
"""
Created on Tue May 15 17:55:48 2018

@author: Wentao Huang
"""
from collections import OrderedDict
import numpy as np
import torch as tc
from .helper import is_tensor, copy_params, flip
from .helper import get_key_val, get_keys_vals, raise_err
from .methods import get_outsize_by_eigv

###############################################################################
PARAMS_0 = OrderedDict()
PARAMS_0['eps'] = 1e-6
PARAMS_0['seed'] = 1
PARAMS_0['iszca'] = False
PARAMS_0['iscent'] = False
PARAMS_0['ismean'] = True
PARAMS_0['denoise'] = 0 # 0, -1, -2, 1, 2, ....
PARAMS_0['energy_ratio'] = 1.0#.99
PARAMS_0['isneg_avg'] = 0.0


def PCA(X, eigenvectors=True, eps=1e-6):
    is_tensor(X, is_raised=True, err="PCA: X", isnone=False)
    mean = tc.mean(X, 0)
    X = X - mean
    Num, K = X.shape
    xx = tc.mm(X.t(), X)/Num
    if not eigenvectors:
        d, U = tc.symeig(xx, eigenvectors=False)
        d[d<eps] = eps
        d = flip(d)
        return d, mean
    else:
        d, U = tc.symeig(xx, eigenvectors=True)
        d[d<eps] = eps
        d = flip(d)
        U = flip(U,1)
        return U, d, mean


def whiten_filter(X, isfull=False, params=None, **kwargs):
    is_tensor(X, is_raised=True, err="whiten_filter: X", isnone=False)
    p = copy_params(is_raised=False, err="whiten_filter", params=params, **kwargs)
    keyslist = ['iszca', 'iscent', 'ismean', 'batch_size', 'denoise',
                'energy_ratio', 'eps', 'seed', 'whiten_lambda']
    assignlist = [False, False, True, 1e+5, 0, 1.0, 1e-15, 1, None]
    [iszca, iscent, ismean, batch_size, denoise, engr,
     eps, seed, whiten_lambda] = get_keys_vals(keyslist, assignlist, params=p)
    dim = X.dim()
    if dim == 1:
        return whiten_filter_0(X, eps)
    raise_err(dim!=2, "whiten_filter: X has %d dimensions but it must be equal to 2"%(dim))
    Num, K = X.shape
    if K == 1:
        return whiten_filter_0(X)
    if iscent:
        xv = X.mean(1, keepdim=True)
        X = X - xv
    if ismean:
        mean = X.mean(0)
        X = X - mean
    else:
        mean = 0.0
    batch_size = int(batch_size)
    xx = 0.0
    if Num <= batch_size:
        xx = (1.0/Num)*tc.mm(X.t(), X)
    else:
        for i in range(0, Num, batch_size):
            m = i + batch_size
            if (Num < m):
                m = Num
            Xi = X[i:m, :]
            xx = xx + (1.0/Num)*tc.mm(Xi.t(), Xi)
            if m >= Num:
                break
    Xi = None
    d, U = tc.symeig(xx, eigenvectors=True)
    xx = None
    d[d<eps] = eps
    d = flip(d)
    U = flip(U,1)
    S = tc.sqrt(d)#+eps
    K0 = get_outsize_by_eigv(d, engr, denoise, K)
    if (denoise == -3):
        S[K0:] = S[K0-1]
        K0 = K
    elif isfull and (whiten_lambda is not None):
        if whiten_lambda < 0.0:
            whiten_lambda = 0.0
        elif whiten_lambda > 1.0:
            whiten_lambda = 1.0
        lambda0 = whiten_lambda * S[0]
        S = S + lambda0
    S[S<eps] = eps
    V = U[:,0:K0]*(1.0/(S[0:K0]))#+eps
    if K0 == 1:
        V = V.reshape((K, 1))
    if iszca:
        V = tc.mm(V, U[:,0:K0].t())
        if K0 == 1:
            V = V.reshape((K, 1))
    X = tc.mm(X, V)
    if K0 == 1:
        X = X.reshape((Num, 1))
    if not isfull:
        U = U[:,:K0]
        if K0 == 1:
            U = U.reshape((K, 1))
        S = S[:K0]
    return X, U, S, mean, K0


def whiten_filter_0(X, eps=1e-6):
    is_tensor(X, is_raised=True, err="whiten_filter_0: input", isnone=False)
    dim = X.dim()
    if dim == 1:
        Num, K0 = X.shape[0], 1
    elif dim == 2:
        Num, K0 = X.shape
    else:
        raise_err(True, "whiten_filter_0: the dimension of input = {} > 2".format(dim))
    raise_err(K0!=1, "whiten_filter_0: K0({}) != 1".format(K0))
    mean = X.mean(0)
    X = X - mean
    S = X.std()
    U = 1.0
    if S < eps:
        S = eps
    X = X/S
    return X, U, S, mean, K0


def whiten_filter_1(X0, X1=None, isfull=False, params=None, **kwargs):
    is_tensor(X0, is_raised=True, err="whiten_filter_1: X0", isnone=False)
    is_tensor(X1, is_raised=True, err="whiten_filter_1: X1", isnone=False)
    p = copy_params(is_raised=False, err="whiten_filter_1", params=params, **kwargs)
    isminusmean = get_key_val(key='isminusmean', assign=True, params=p)
    N0, K = X0.shape
    N1 = X1.shape[0]
    X0, U, S, mean, K0 = whiten_filter(X=X0, isfull=isfull, params=p)
    V = U[:,0:K0]*(1.0/(S[0:K0]))#+eps
    if K0 == 1:
        V = V.reshape((K, 1))
    if isminusmean:
        X1 = tc.mm(X1-mean, V)
    else:
        X0 = X0 + tc.mv(V.t(), mean)
        X1 = tc.mm(X1, V)
        mean = 0.0*mean
    if not isfull:
        U = U[:,:K0]
        if K0 == 1:
            U = U.reshape((K, 1))
        S = S[:K0]
    if K0 == 1:
        X0 = X0.reshape((N0, 1))
        X1 = X1.reshape((N1, 1))
    return X0, X1, U, S, mean, K0

def whiten_filter_2(X0, X1=None, isfull=False, params=None, **kwargs):
    is_tensor(X0, is_raised=True, err="whiten_filter_2: X0", isnone=False)
    is_tensor(X1, is_raised=True, err="whiten_filter_2: X1", isnone=False)
    p = copy_params(is_raised=False, err="whiten_filter_2", params=params, **kwargs)
    dim = X0.dim()
    if dim != 2:
        raise ValueError("whiten_filter_2: X has {} dimensions "
                         "but it must be equal to 2".format(dim))
    keyslist = ['iszca', 'iscent', 'ismean', 'balance_factor',
                'denoise', 'energy_ratio', 'eps', 'isneg_avg',
                'seed', 'batch_size', 'whiten_lambda']
    assignlist = [False, False, True, None, 0, 1.0, 1e-15, False, 1, 1e+5, None]
    [iszca, iscent, ismean, balance_factor, denoise, engr, eps,isneg_avg, seed,
     batch_size, whiten_lambda] = get_keys_vals(keyslist, assignlist, params=p)
    N0, K = X0.shape
    N1, K1 = X1.shape
    if K != K1:
        raise ValueError("whiten_filter_2: K != K1, ({}, {})".format(K, K1))
    if iscent:
        xv = X0.mean(1).reshape((N0, 1))
        X0 = X0 - xv
        xv = X1.mean(1).reshape((N1, 1))
        X1 = X1 - xv
    if ((balance_factor is None) or (balance_factor < 0.0)
                                or (balance_factor > 1.0)):
        factor = N0/(N0 + N1)
    else:
        factor = balance_factor
    if ismean:
        s0 = X0.mean(0)
        s1 = X1.mean(0)
        if isneg_avg:
            mean = factor*s0 - (1.0-factor)*s1
        else:
            mean = factor*s0 + (1.0-factor)*s1
        X0 = X0 - mean
        X1 = X1 - mean
    else:
        mean = 0.0
    batch_size = int(batch_size)
    xx0 = 0.0
    if N0 <= batch_size:
        xx0 = (1.0/N0)*tc.mm(X0.t(), X0)
    else:
        for i in np.arange(0, N0, batch_size):
            m = i + batch_size
            if (N0 < m):
                m = N0
            Xi = X0[i:m, :]
            xx0 = xx0 + (1.0/N0)*tc.mm(Xi.t(), Xi)
            if m >= N0:
                break
    xx1 = 0.0
    if N1 <= batch_size:
        xx1 = (1.0/N1)*tc.mm(X1.t(), X1)
    else:
        for i in np.arange(0, N1, batch_size):
            m = i + batch_size
            if (N1 < m):
                m = N1
            Xi = X1[i:m, :]
            xx1 = xx1 + (1.0/N1)*tc.mm(Xi.t(), Xi)
            if m >= N1:
                break
    xx = factor*xx0 + (1.0 - factor)*xx1
    Xi = xx1 = xx0 = None
    d, U = tc.symeig(xx, eigenvectors=True)
    xx = None
    d[d<eps] = eps
    d = flip(d)
    U = flip(U,1)
    S = tc.sqrt(d)#+eps
    K0 = get_outsize_by_eigv(d, engr, denoise, K)
    if (denoise == -3):
        S[K0:] = S[K0-1]
        K0 = K
    elif isfull and (whiten_lambda is not None):
        if whiten_lambda < 0.0:
            whiten_lambda = 0.0
        elif whiten_lambda > 1.0:
            whiten_lambda = 1.0
        lambda0 = whiten_lambda * S[0]
        S = S + lambda0
    S[S<eps] = eps
    V = U[:,0:K0]*(1.0/(S[0:K0]))#+eps
    if K0 == 1:
        V = V.reshape((K, 1))
    if iszca:
        V = tc.mm(V, U[:,0:K0].t())
        if K0 == 1:
            V = V.reshape((K, 1))
    X0 = np.dot(X0, V)
    X1 = np.dot(X1, V)
    if K0 == 1:
        X0 = X0.reshape((N0, 1))
        X1 = X1.reshape((N1, 1))
    if not isfull:
        U = U[:,:K0]
        if K0 == 1:
            U = U.reshape((K, 1))
        S = S[:K0]
    return X0, X1, U, S, mean, K0
