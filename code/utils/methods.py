# methods.py
"""
Created on Mon May  7 10:38:54 2018

@author: Wentao Huang
"""
from __future__ import absolute_import, division, print_function
#import ipdb
import torch as tc
from .helper import is_tensor, raise_err, flip


def Gram_Schmidt_Orth_Col_0(C, eps=1e-6):
    r"""
    Classical Gram-Schmidt Orthogonalization for lolumn vectors
    """
    is_tensor(C, is_raised=True, err="Gram_Schmidt_Orth_Col: input", isnone=False)
    shape = C.size()
    if (len(shape)!= 2):
        raise ValueError("Gram_Schmidt_Orth_Col: the dimension of input is %d" %(len(shape)))
    K, KA = shape
    if (K < KA):
        raise ValueError("Gram_Schmidt_Orth_Col: the shape of input = (%d, %d)" %(K, KA))
#    ipdb.set_trace()
    U = tc.zeros(K, KA)
    u = C[:, 0]
    U[:, 0] = u.div_(u.norm(2) + eps)
    for i in range(1, KA):
        ui = C[:, i]
        for j in range(i):
            uj = U[:,j]
            ui = ui - (ui.dot(uj)/(uj.dot(uj)+eps))*uj
        U[:, i] = ui.div_(ui.norm(2) + eps)
    return C


def Gram_Schmidt_Orth_Col(C, eps=1e-6):
    r"""
    Modified Gram-Schmidt Orthogonalization for lolumn vectors
    """
    is_tensor(C, is_raised=True, err="Gram_Schmidt_Orth_Col: input", isnone=False)
    shape = C.size()
    if (len(shape)!= 2):
        raise ValueError("Gram_Schmidt_Orth_Col: the dimension of input is %d" %(len(shape)))
    K, KA = shape
    if (K < KA):
        raise ValueError("Gram_Schmidt_Orth_Col: the shape of input = (%d, %d)" %(K, KA))
    for i in range(KA):
        ci = C[:, i]
        C[:, i] = ci.div_(ci.norm(2) + eps)
        for j in range(i+1,KA):
            cj = C[:,j]
            C[:,j] = cj - ci.dot(cj)*ci
    return C


def Gram_Schmidt_Orth_Row(C, eps=1e-6):
    r"""
    Modified Gram-Schmidt Orthogonalization for row vectors
    """
    is_tensor(C, is_raised=True, err="Gram_Schmidt_Orth_Row: input", isnone=False)
    shape = C.size()
    if (len(shape)!= 2):
        raise ValueError("Gram_Schmidt_Orth_Row: the dimension of input is %d" %(len(shape)))
    K, KA = shape
    if (K > KA):
        raise ValueError("Gram_Schmidt_Orth_Row: the shape of input = (%d, %d)" %(K, KA))
    for i in range(K):
        ci = C[i, :]
        C[i, :] = ci.div_(ci.norm(2) + eps)
        for j in range(i+1,K):
            cj = C[j,:]
            C[j,:] = cj - ci.dot(cj)*ci
    return C


def Gram_Schmidt_Orth(C, isrow=False, eps=1e-6): # Orthogonalization for row vectors or column vectors
    if isrow:
        return Gram_Schmidt_Orth_Row(C, eps)
    else:
        return Gram_Schmidt_Orth_Col(C, eps)


def init_weights0(in_features, out_features, seed=1, extent=None):
    import math
    if extent is None:
        extent = 1.0/math.sqrt(in_features)
    W = tc.Tensor(out_features, in_features)
    tc.manual_seed(seed)
    return W.data.uniform_(-extent, extent)


def init_weights(in_features, out_features, seed=1, extent=None):
    import math
    if extent is None:
        extent = 1.0/math.sqrt(in_features)
    assert extent > 0.0
    tc.manual_seed(seed)
    W = tc.rand((out_features, in_features))#n.mul_(extent)
    W.add_(-0.5).mul_(2.0*extent)
    return W

def pre_weights(W, isorth=True, isnorm=True, isrow=False, eps=1e-6):
    assert tc.is_tensor(W), 'W must be a Tensor type'
    assert W.dim() == 2, 'W.dim must be equal to 2'
    K, KA = W.size()
    d = 1 if isrow else 0
    if isorth:
        if KA == 1 or K == 1:
            W = W.div(W.norm(2,dim=d, keepdim=True)+eps)
        else:
            W = Gram_Schmidt_Orth(W, isrow, eps)
    elif isnorm:
        W = W.div(W.norm(2, dim=d, keepdim=True) + eps)
    return W


def get_outsize_by_eigv(eigvalues, rate=None, denoise=0, refsize=None):
    is_tensor(eigvalues, is_raised=True, err="get_outsize_by_eigv: eigvalues", isnone=False)
    if denoise == 0:
        return int(len(eigvalues))
    raise_err(eigvalues is None, "get_outsize_by_eigv: eigvalues is None")
#    ipdb.set_trace()
    if denoise == -1:
        raise_err(rate is None, "get_outsize_by_eigv: rate is None")
        if refsize is None or refsize > len(eigvalues):
            refsize = len(eigvalues)
        dc = tc.sqrt(tc.cumsum(eigvalues,dim=0)/tc.sum(eigvalues))
        K0 = int(tc.sum(dc<=rate)) # e.g. rate = 0.99
        K0 = refsize if refsize < K0 else K0 #tc.max(tc.tensor([K0+1, refsize]))
    elif denoise == -2:
        raise_err(rate is None, "get_outsize_by_eigv: rate is None")
        if refsize is None or refsize > len(eigvalues):
            refsize = len(eigvalues)
        dc = tc.sqrt(tc.cumsum(eigvalues,dim=0)/tc.sum(eigvalues))
        K0 = int(tc.sum(dc<=rate)) # e.g. rate = 0.99
        K0 = refsize if refsize > K0 else K0 #tc.max(tc.tensor([K0+1, refsize]))
    elif denoise == 1:
        if refsize is None or refsize > len(eigvalues):
            refsize = len(eigvalues)
        if rate is None:
            K1 = len(eigvalues)
        else:
            dc = tc.sqrt(eigvalues/eigvalues[0])
            K1 = int(tc.sum(dc>=rate)) # e.g. rate = 1e-2
        K0 = K1 if K1 < refsize else refsize #tc.min(tc.tensor([K1, round(denoise)]))
    elif denoise == 2:
        if refsize is None or refsize > len(eigvalues):
            refsize = len(eigvalues)
        if rate is None:
            K1 = len(eigvalues)
        else:
            dc = tc.sqrt(eigvalues/eigvalues[0])
            K1 = int(tc.sum(dc>=rate)) # e.g. rate = 1e-2
        K0 = K1 if K1 > refsize else refsize
    elif denoise == 3:
        dc = tc.sqrt(eigvalues/eigvalues[0])
        K0 = int(tc.sum(dc>=rate))# e.g. rate = 1e-2
    elif refsize is not None:
        K0 = int(refsize)
    else:
        raise ValueError("get_outsize_by_eigv: inputs error")
    K0 = K0 if K0 > 1 else 1
    K0 = K0 if K0 <= int(len(eigvalues)) else int(len(eigvalues))
    return K0


def get_outsize(refsize=100, outsize=1, outscale=1, out_zoom=0):
    if out_zoom == 0:
        return int(refsize)
    if out_zoom < 0:
        if outsize < refsize:
            outsize0 = outsize
        elif outscale < 1:
            outsize0 = round(refsize*outscale)
        else:
            outsize0 = refsize
    else:
        if outsize > refsize:
            outsize0 = outsize
        elif outscale > 1:
            outsize0 = round(refsize*outscale)
        else:
            outsize0 = refsize
    outsize0 = outsize0 if outsize0 > 1 else 1
    return int(outsize0)


def get_filters(C, U=None, iszca=False):
    K, KA = C.size()
    if not iszca:
        C = U[:,:K].mm(C)
    return C.t()


def get_filters2(C, U=None, U0=None, iszca=False):
    if U0 is not None:
        C = U0.mm(C)
    K, KA = C.size()
    if not iszca:
        C = U[:,:K].mm(C)
    return C.t()


def get_bases(C, U=None, S=None, iszca=False, islesq=False):
    K, KA = C.shape
    if iszca and U is not None:
        C = U.t().mm(C)
    if islesq and K > KA:
        I = tc.diag(tc.ones(K))
        Cv, _ = tc.gels(I, C)
    else:
#        ipdb.set_trace()
        u, s, v = tc.svd(C, some=True)
        k0 = tc.sum(s/s[0]>=1e-2)
        Cv = u[:,:k0]*(1.0/s[:k0])
        Cv = Cv.mm(v[:,:k0].t())
    if U is None or S is None:
        B = Cv.t()
    else:
        V = U[:,:K]*S[:K]
        B = V.mm(Cv).t()
    return B


def get_bases2(C, U=None, S=None, U0=None, iszca=False, islesq=False):
    if U0 is not None:
        C = U0.mm(C)
    K, KA = C.shape
    if iszca and U is not None:
        C = U.t().mm(C)
    if islesq and K > KA:
        I = tc.diag(tc.ones(K))
        Cv, _ = tc.gels(I, C)
    else:
        u, s, v = tc.svd(C, some=True)
        k0 = tc.sum(s/s[0]>=1e-2)
        Cv = u[:,:k0]*(1.0/s[:k0])
        Cv = Cv.mm(v[:,:k0].t())
    if U is None or S is None:
        B = Cv.t()
    else:
        V = U[:,:K]*S[:K]
        B = V.mm(Cv).t()
    return B


def sort_weights(W, index=False):
    shape = W.size()
    W = W.view(shape[0], -1)
    std = W.std(1)
    std, ind = tc.sort(std, descending=True)
    W = W[ind,:].view(shape)
    if index:
        return W, ind
    return W
    

def get_diag_e(E0, C, eps=1e-6):
    K, KA = C.size()
#    print('K={}, KA={}, E.size={}'.format(K, KA, E0.size(0)))
#    ipdb.set_trace()
    if E0 is None:
        e0 = e1 = eps
    elif tc.is_tensor(E0):
        if K == KA:
            e0 = eps/E0
            e1 = eps/(1.0-E0)
        else:
            if KA == 1:
                e0, _ = tc.symeig((C.t()*(eps/E0)).mm(C), eigenvectors=True)
                e1, _ = tc.symeig((C.t()*(eps/(1.0-E0))).mm(C), eigenvectors=True)
#                e0 = (C.t()*(E0)).mm(C).squeeze(0)
#                e0[e0<1e-8] = 1e-8
#                e0 = eps/e0
#                e1 = (C.t()*(1.0-E0)).mm(C).squeeze(0)
#                e1[e1<1e-8] = 1e-8
#                e1 = eps/e1
            else:
                e0, _ = tc.symeig((C.t()*(eps/E0)).mm(C), eigenvectors=True)
                e1, _ = tc.symeig((C.t()*(eps/(1.0-E0))).mm(C), eigenvectors=True)
#                e0, _ = tc.symeig((C.t()*(E0)).mm(C), eigenvectors=True)
#                e1, _ = tc.symeig((C.t()*((1.0-E0))).mm(C), eigenvectors=True)
#                e0[e0<1e-8] = 1e-8
#                e1[e1<1e-8] = 1e-8
#                e0 = eps/e0
#                e1 = eps/e1
                e1 = flip(e1)
    else:
        e0 = e1 = E0
    return e0, e1
    

def set_diag_e(e, E, E0, C, eps=1e-6):
#    ipdb.set_trace()
    if E is not None:
        e = E
        print('e = {:.2e}, E = {:.2e}'.format(e, E))
    elif tc.is_tensor(E0):
        if C.size(1) == 1:
            e0, _ = tc.symeig((C.t()*(eps/E0)).mm(C), eigenvectors=True)
#            e0 = (C.t()*(E0)).mm(C).squeeze(0)
#            e0[e0<1e-8] = 1e-8
#            e0 = eps/e0
        else:
            e0, _ = tc.symeig((C.t()*(eps/E0)).mm(C), eigenvectors=True)
#            e0[e0<1e-8] = 1e-8
#            e0 = eps/e0
#            e0 = flip(e0)
        if e is None:
            e = e0
        else:
            e = tc.cat((e, e0))
        print('e0 = [{:.2e}, {:.2e}], E0 = [{:.2e}, {:.2e}]'.format(
              e0[0], e0[-1], E0[0], E0[-1]))
    else:
        e = eps
    return e

    