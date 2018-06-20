# constraint.py
"""
Created on Wed Jun  6 19:05:42 2018

@author: Wentao Huang
"""
#import ipdb
#import torch as tc

def const_lptanh(X, f, df, center, Fc, dFc, ith_cls=0, bias_requires_grad=True, 
                 balance_factor=0.5, beta0=1.0, eps=1e-6):
    assert isinstance(Fc, list) and isinstance(dFc, list)
    N0 = f.size(0)
    M = len(Fc)
    assert ith_cls >=0 and ith_cls < M
    R = [(1.0-balance_factor)/(M-1)/N0]*M
    R[ith_cls] = balance_factor/N0
    obj = 0.0
    dQ = 0.0
    db = 0.0
    for i, fc in enumerate(Fc):
        r = R[i]
        if r == 0.0:
            continue
#        ipdb.set_trace()
        for m in range(fc.size(0)):
            v = fc[m]
            f = f.add_(-v).mul_(beta0).tanh_()
            f2 = f.pow(2)
            if i == ith_cls:
                s = 1.0
                z = eps + 2.0 - f2
            else:
                s = -1.0
                z = f2.add(eps)
            f.mul_(f2.add_(-1.0))
            f0 = f.mul(dFc[i][m]).div_(z).sum(0)#.view(1,-1)
            f.mul_(df).div_(z)
            dQ += -2.0*s*r*beta0*(X.t().mm(f) - center[i][m].view(-1,1).mm(f0.view(1,-1)))
            obj += -r*z.log_().sum()
            if bias_requires_grad:
                db += -2.0*s*r*beta0*(f.sum(0) - f0)
    return obj, dQ, db
