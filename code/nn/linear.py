# linear.py
"""
Created on Sat May  5 15:50:37 2018

@author: Wentao Huang
"""
from copy import deepcopy
import torch as tc
from .base import Base
from ..utils.helper import copy_params, to_Param, get_keys_vals
from ..utils.data import OrderedDataset
from ..utils.methods import init_weights


class Linear(Base):
    r"""Applies a linear transformation to the incoming data: :math:`y = beta*(coff_a*Wx + b - m)`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: coefficient b. If set to False, the layer will not learn an additive bias.
              Default: 0.0
        weight: coefficient W
        coff_a: coefficient coff_a
        beta: coefficient beta
        margin: coefficient m
        is_learnt: coefficient is required to learn
                if is_learnt = True. Default: True.

        training_model: model for training. Default: 'linear'
        dtype: data type

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features=None,
                 out_features=None,
                 bias=None, weight=None,
                 coff_a=None, beta=None,
                 margin=None, requires_grad=True,
                 dtype=None, name='Linear'):
        self.reinit(in_features, out_features, bias, weight, coff_a, 
                    beta, margin, requires_grad, dtype, name)
        
    def reinit(self, in_features=None,
               out_features=None,
               bias=None, weight=None,
               coff_a=None, beta=None, margin=None, 
               requires_grad=True, dtype=None, name='Linear'):
        super(Linear, self).reinit()
        self.set_name(name)
        self.set_parameters(in_features, out_features, bias, weight,
                            coff_a, beta, margin, requires_grad, dtype)

    def set_parameters(self, in_features=None,
                       out_features=None,
                       bias=None, weight=None,
                       coff_a=None, beta=None, margin=None, 
                       requires_grad=True, dtype=None):
        self.set_in_out_features(in_features, out_features)
        self.set_weight(weight, requires_grad, dtype)
        self.set_bias(bias, False, dtype)
        self.set_margin(margin, dtype)
        self.set_coff_a(coff_a, False, dtype)
        self.set_beta(beta, dtype)
        
    def set_in_out_features(self, in_features=None, out_features=None):
        if in_features is None:
            self.in_features = None
        else:
            if in_features <= 0:
                raise ValueError('{}: in_features must be greater than zero'.format(self._name))
            self.in_features = round(in_features)
        if out_features is None:
            self.out_features = None
        else:
            if out_features <= 0:
                raise ValueError('{}: out_features must be greater than zero'.format(self._name))
            self.out_features = round(out_features)
        return self.in_features, self.out_features
        
    def set_weight(self, weight=None, requires_grad=True, dtype=None):
        if weight is None:
            if self.in_features is not None and self.out_features is not None:
                weight = init_weights(self.in_features, self.out_features)
                weight = to_Param(weight.t(), requires_grad, dtype)
            else:
                self.out_features = None
                self.in_features = None
        else:
            weight = to_Param(weight, requires_grad, dtype)
            s = weight.size()
            ls = len(s)
            if ls == 0 or s[0] == 0:
                raise ValueError('The length of shape of weight cannot be zero!')
            elif ls==1:
                weight = weight.veiw(1,s[0])#weight.unsqueeze(0)
            s = weight.size()
            self.out_features = s[0]
            m = 1
            for i in s[1:]:
                m *= i
            self.in_features = m
        self.register_parameter('weight', weight)
        return weight
    
    def reset_weight(self, weight=None, requires_grad=True, dtype=None):
        if weight is None:
            weight = self.weight
        return self.set_weight(weight, requires_grad, dtype)
        
    def set_bias(self, bias=None, requires_grad=False, dtype=None):
        if bias is None or bias is False:
            bias = None
        elif bias is True:
            bias = to_Param([0.0], True, dtype)
        else:
            bias = to_Param(bias, requires_grad, dtype)
        self.register_parameter('bias', bias)
        return bias

    def reset_bias(self, bias=None, requires_grad=False, dtype=None):
        if bias is None:
            bias = self.bias
        return self.set_bias(bias, requires_grad, dtype)
    
    def set_margin(self, margin=None, dtype=None):
        if margin is None or margin is False:
            margin = None
        elif margin is True:
            margin = to_Param(0.0, False, dtype)
        else:
            margin = to_Param(margin, False, dtype)
        self.register_parameter('margin', margin)
        return margin

    def reset_margin(self, margin=None, dtype=None):
        if margin is None:
            margin = self.margin
        return self.set_margin(margin, dtype)
    
    
    def set_beta(self, beta=None, dtype=None):
        if beta is None or beta is False:
            beta = None
        else:
            beta = to_Param(beta, False, dtype)
        self.register_parameter('beta', beta)
        return beta

    def reset_beta(self, beta=None, dtype=None):
        if beta is None:
            beta = self.beta
        return self.set_beta(beta, dtype)
    
    def set_coff_a(self, coff_a=None, requires_grad=False, dtype=None):
        if coff_a is None or coff_a is False:
            coff_a = None
        else:
            coff_a = to_Param(coff_a, requires_grad, dtype)
        self.register_parameter('coff_a', coff_a)
        return coff_a

    def reset_coff_a(self, coff_a=None, requires_grad=False, dtype=None):
        if coff_a is None:
            coff_a = self.coff_a
        return self.set_coff_a(coff_a, requires_grad,  dtype)
    
    def linear(self, input, weight, bias=None, iscent=False):
        """
        Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    
        Shape:
            - Input: :math:`(N, *, in\_features)` where `*` means any number of
              additional dimensions
            - Weight: :math:`(out\_features, in\_features)`
            - Bias: :math:`(out\_features)`
            - Output: :math:`(N, *, out\_features)`
        """
#        print('{}.linear: iscent = {}'.format(self.name, iscent))
        if iscent:
            input = input - input.mean(1, keepdim=True)
        if input.dim() == 2 and bias is not None:
            # fused op is marginally faster
            return tc.addmm(bias, input, weight.t())
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    def ordlinear(self, input, weight, bias=None, iscent=False):
        assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
        output = []
        cat_end = input.cat_end
        max_iter_samples = input.max_iter_samples
        input.set_sample_iter(0, True)
#        print('{}.ordlinear: iscent = {}'.format(self.name, iscent))
        for X, _ in input:
            if iscent:
                X = X - X.mean(1, keepdim=True)
            output.append(self.linear(X, weight, bias))
        output = OrderedDataset(output, max_iter_samples=max_iter_samples, cat_end=cat_end)
        output.cls_labels = deepcopy(input.cls_labels)
        input.set_sample_iter(max_iter_samples, cat_end)
        return output
    
    def forward(self, input=None):
        if input is None:
            input = self.input
        if input is None:
            return None
        if self.weight is None:
            raise ValueError('{}: _parameters[\'weight\'] is None'.format(self.name))
        weight, bias = self.get_weight_bias(self.weight, self.bias, self.beta, 
                                            self.margin, self.coff_a)
        if isinstance(input, tc.Tensor):
            return self.linear(input, weight, bias)
        elif isinstance(input, OrderedDataset):
            return self.ordlinear(input, weight, bias)
        else:
            return None
    
    def extra_repr(self):
        if self.in_features is None:
            in_features = ''
        else:
            in_features = 'in_features={}, '.format(self.in_features)
        if self.out_features is None:
            out_features = ''
        else:
            out_features = 'out_features={}, '.format(self.out_features)
        if tc.is_tensor(self.bias):
            bias = 'bias={:.2e}, '.format(self.bias.mean().item())
        else:
            bias = ''
        if tc.is_tensor(self.coff_a):
            coff_a = 'coff_a={:.2f}, '.format(self.coff_a.item())
        else:
            coff_a = ''
        margin = 'margin={:.2f}, '.format(self.margin.item()) if self.margin is not None else ''
        beta = 'beta={:.2f}, '.format(self.beta.item()) if self.beta is not None else ''
        str_print = in_features+out_features+bias+beta+margin+coff_a+'name={}'.format(self.name)
        return str_print
    
    @staticmethod
    def get_weight_bias(weight=None, bias=None, beta=None, 
                        margin=None, coff_a=None):
        if weight is not None:
            if coff_a is not None:
                weight = coff_a*weight
        else:
            if coff_a is not None:
                weight = coff_a
        if beta is not None:
            if weight is not None:
                weight = beta*weight
            else:
                weight = beta
            if bias is not None:
                if margin is not None:
                    bias = beta*(bias - margin)
                else:
                    bias = beta*bias
            else:
                if margin is not None:
                    bias = - beta*margin
        else:
            if bias is not None:
                if margin is not None:
                    bias = bias - margin
            else:
                if margin is not None:
                    bias = - margin
        return weight, bias

    @staticmethod
    def predata(input=None, params=None, **kwargs):
        assert isinstance(input, OrderedDataset), 'input must be an OrderedDataset type'
        p = copy_params(is_raised=False, params=params, **kwargs)
        keyslist = ['iscent', 'isstd', 'ismean', 'isnorm', 'inplace', 
                    'batch_size', 'eps']
        assignlist = [True, True, True, False, True, None, 1e-10]
        [iscent, isstd, ismean, isnorm, inplace, 
         batch_size, eps] = get_keys_vals(keyslist, assignlist, params=p)
        if batch_size is None or batch_size < 0:
            batch_size = 0
        max_iter_samples = input.max_iter_samples
        cat_end = input.cat_end
        input.set_sample_iter(batch_size, True)
        shape = input.size()[1:]
        dim = input.dim()
        if dim > 2:
            input.view_(-1)
        N = len(input)
        mean = 0.0
        input.reiter()
        for X, name in input:
            if iscent:
                X = X.add(-X.mean(1, keepdim=True))
            if isstd:
                X = X.mul(1.0/X.std(1, keepdim=True))
            if ismean:
                mean = mean + 1.0/N*X.sum(0)
        input.reiter()
        if inplace:
            for X, name in input:
                if iscent:
                    X.add_(-X.mean(1, keepdim=True))
                if isstd:
                    X.div_(X.std(1, keepdim=True)+eps)
                if ismean:
                    X.add_(-mean.view(1,-1))
                if isnorm:
                    X.div_(X.norm(2, dim=1, keepdim=True)+eps)
                input.set_sample_iter(max_iter_samples, cat_end)
                if dim > 2:
                    input.view_(shape)
                return input
        else:
            input.set_sample_iter(0, True)
            output = []
            for X, name in input:
                if iscent:
                    X = X.add(-X.mean(1, keepdim=True))
                if isstd:
                    X = X.div(X.std(1, keepdim=True)+eps)
                if ismean:
                    X = X.add(-mean.view(1,-1))
                if isnorm:
                    X = X.div(X.norm(2,dim=1,keepdim=True)+eps)
                output = OrderedDataset(output, cat_end=cat_end,
                                        max_iter_samples=max_iter_samples)
                output.cls_labels = deepcopy(input.cls_labels)
                input.set_sample_iter(max_iter_samples, cat_end)
                if dim > 2:
                    output.view_(shape)
                    input.view_(shape)
                return output
