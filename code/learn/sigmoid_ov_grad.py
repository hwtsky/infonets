# sigmoid_ov_grad.py
"""
Created on Fri May 25 11:23:23 2018

@author: Wentao Huang
"""
#import ipdb
import torch as tc
from .grad import Grad

class SigmoidOVGrad(Grad):

    @staticmethod
    def forward(ctx, input, C, bias=None, beta=1.81, 
                isorth=True, eps=1e-6):
        if isinstance(bias, tc.Tensor):
            bias_requires_grad = bias.requires_grad
        else:
            bias_requires_grad = False
        if bias is not None:
            bias = beta * bias
        K0, KA = C.size()
        assert K0 <= KA
        rc = tc.sqrt(tc.tensor(1.0*KA/K0))
        C1 = beta * C
        Num = len(input)
        input.reiter()
        obj = 0.0
        db = 0.0 if bias_requires_grad else None#tc.Tensor([0.0])
        dC = 0.0
        for X, _ in input:
#            ipdb.set_trace()
            Ni = X.size(0)
            f = X.mm(C1)
            if bias is not None:
                f.add_(bias)
            f.sigmoid_()
            gd = f.pow(2).mul_(-1.0).add_(f) #f*(1.0 - f)
            gdv = gd.mean(0)
            gd.mul_(f.mul_(-2.0).add_(1.0))
            xgdd = rc*beta/Num*X.t().mm(gd)
            
            Cg = C*gdv
            G = Cg.mm(Cg.t()) + tc.diag(tc.full((K0,), eps))
            sign, logdet = tc.slogdet(G)
            obj = obj - 0.5*Ni/Num*logdet
            
            GC, LU = tc.gesv(C, G)
            gv = (GC*C).sum(0)*gdv
            dC = dC -GC*(gdv*gdv) - xgdd*gv
            
            if bias_requires_grad:
                db0 = beta/Num*gd.sum(0)
                db = db - db0*gv
        dC = C.mm(C.t()).mm(dC)
        if isorth:
            dC = dC - C.mm(dC.t()).mm(C)
        argnum = tc.tensor([6])
        ctx.save_for_backward(dC, db, argnum)
#        ipdb.set_trace()
        return obj
