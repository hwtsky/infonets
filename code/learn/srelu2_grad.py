# srelu2_grad.py
"""
Created on Thu May 31 12:49:01 2018

@author: Wentao Huang
"""
import ipdb
import torch as tc
from .grad import Grad
from ..utils.methods import get_diag_e

class SReLU2Grad(Grad):

    @staticmethod
    def forward(ctx, input, C, bias=None, beta=1.0, isorth=True, eps=1e-6, 
                ith_cls=0, R=None, margin=None, E0=None):
        if isinstance(bias, tc.Tensor):
            bias_requires_grad = bias.requires_grad
        else:
            bias_requires_grad = False
        if bias is not None:
            bias = beta * bias
        else:
            bias = 0.0
        if margin is not None:
            margin = beta * margin
        else:
            margin = 0.0
        K, KA = C.size()
        C1 = beta * C
        e0, e1 = get_diag_e(E0, C1, eps)
#        ipdb.set_trace()
#        N0 = input.get_len(ith_cls)
#        N1 = len(input) - N0
#        R = input.get_len_list()
#        M = len(R)
#        if balance_factor == 0.5:
#            R = [0.5/((M-1)*i) for i in R]
#            R[ith_cls] = 0.5/N0
#        else:
#            R = [(1.0-balance_factor)/N1]*M
#            R[ith_cls] = balance_factor/N0
        input.reiter()
#        cls_label = input.get_label(ith_cls)
        obj0 = 0.0
        obj1 = 0.0
        db = 0.0 if bias_requires_grad else None#tc.Tensor([0.0])
        dQ = 0.0
        for X, label in input:
#            i = input.get_index(label)
            r = R[label]
            if label == ith_cls:
                if r == 0.0:
                    continue
                s = 1.0
                e = e0
                f = X.mm(C1)
                bias0 = bias - margin
            else:
                if r == 0.0:
                    continue
                s = -1.0
                e = e1
                f = X.mm(C1.mul(-1.0))
                bias0 = -bias - margin
            if bias0 is not 0.0:
                f.add_(bias0)
            f.sigmoid_()
            f2 = f.pow(2)
            g = f2.add(e)
#            ipdb.set_trace()
            f2.mul_(1.0-f).div_(g)
            obj0 += -0.5*r*g.log_().sum()
            dQ = dQ - s*r*beta*X.t().mm(f2)
            if bias_requires_grad:
                db = db - s*r*beta*f2.sum(0)
        KA = C.size(1)
        if KA == 1:
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
        argnum = tc.tensor([10])
        ctx.save_for_backward(dC, db, argnum)
        return obj0 + obj1
