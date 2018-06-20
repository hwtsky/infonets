# lpsrelu_const_grad.py
"""
Created on Thu Jun  7 12:22:38 2018

@author: Wentao Huang
"""
import ipdb
import torch as tc
from .grad import Grad
from . import constraint

class LPSReLUConstGrad(Grad):

    @staticmethod
    def forward(ctx, input, C, bias=None, beta=3.0, 
                isorth=True, eps=1e-6, const_fun='const_lptanh', 
                margin=None, beta0=0.1, const_factor=0.1, 
                balance_factor=0.5, alpha=0.8, center=None):
        F = getattr(constraint, const_fun, None)
        if isinstance(bias, tc.Tensor):
            bias_requires_grad = bias.requires_grad
        else:
            bias_requires_grad = False
        if bias is not None:
            if margin is not None:
                bias = bias - margin
        else:
            if margin is not None:
                bias = -margin
        if bias is not None:
            bias = beta * bias
        C1 = beta * C
        Fc = []
        dFc = []
        if center is not None:
            assert isinstance(center, list)
#            ipdb.set_trace()
            for f in center:
                f = f.mm(C1)
                if bias is not None:
                    f.add_(bias)
                Fc.append(f.exp().add_(1.0).log_().mul_(alpha/beta).add_((1.0-alpha)/beta*f))
                dFc.append(f.sigmoid().mul_(alpha).add_(1.0-alpha+eps))
        R = input.get_len_list()
        M = len(R)
        if balance_factor == 0.5:
            R = [0.5/(M*i) for i in R]
        else:
            R = [1.0/len(input)]*M
        r0 = const_factor/M
        input.reiter()
        obj0 = 0.0
        db = 0.0 if bias_requires_grad else None#tc.Tensor([0.0])
        dQ = 0.0
        for X, name in input:
            ith_cls = input.get_index(name)
            r = R[ith_cls]#0.0#
#            ipdb.set_trace()
            f = X.mm(C1)
            if bias is not None:
                f.add_(bias)
            f0 = f.exp().add_(1.0).log_().mul_(alpha/beta).add_((1.0-alpha)/beta*f)
            f.sigmoid_()
            g = f.mul(alpha).add_(1.0-alpha+eps)
            f.mul_(1.0-f).div_(g)#.mul_(alpha*beta)
            if r0 > 0.0:
                objz, dQz, dbz = F(X, f0, g, center, Fc, dFc, ith_cls, 
                                 bias_requires_grad, balance_factor, beta0, eps)
            else:
                objz = dQz = dbz = 0.0
            f0 = None
            obj0 += g.log_().sum().mul_(-r) + r0*objz
            dQ += -r*alpha*beta*(X.t().mm(f)) + r0*dQz
            if bias_requires_grad:
                db += -r*alpha*beta*f.sum(0) + r0*dbz
            f = g = None
        K = C.size(1)
        if K == 1:
            G = C.t().mm(C) + eps
            obj1 = -0.5*G.log()
        else:
            G = C.t().mm(C) + tc.diag(tc.full((C.size(1),), eps))
            sign, logdet = tc.slogdet(G)
            obj1 = -0.5*logdet
        dQ = dQ.mm(C.t().mm(C))
        if isorth:
            dC = dQ - C.mm(dQ.t()).mm(C)
        else:
            dC = dQ - C
        argnum = tc.tensor([13])
        ctx.save_for_backward(dC, db, argnum)
        return obj0 + obj1


