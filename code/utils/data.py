# data.py
"""
Created on Sun May 16 15:30:38 2018

@author: Wentao Huang
"""
from __future__ import absolute_import, division
#import ipdb
from collections import Iterable
from copy import copy, deepcopy
import torch as tc
import torch.utils.data as data


class OrderedDataset(data.Dataset):
    def __init__(self, dataset=None, num_cls=None,
                 dom_cls=None, max_iter_samples=0,
                 cat_end=True, isdeepcopy=False):
        super(OrderedDataset, self).__init__()
        if dataset is None:
            self.cls_list = []
            self.cls_labels = []
            self.dom_cls = dom_cls
            self.sample_iter_list = []
            self.sample_iter = iter(self.sample_iter_list)
            self.max_iter_samples = int(max_iter_samples)
            self.cat_end = bool(cat_end)
        else:
            if isinstance(dataset, OrderedDataset):
                self.copy(dataset, isdeepcopy)
                self.set_sample_iter(max_iter_samples, cat_end)
            elif isinstance(dataset, data.Dataset):
                self.set_dataset(dataset, num_cls)
                self.dom_cls = dom_cls
                self.set_sample_iter(max_iter_samples, cat_end)
            elif isinstance(dataset, list) and len(dataset) > 0:
                for i in range(len(dataset)):
                    assert isinstance(dataset[i], tc.Tensor)
                self.set_cls_list(dataset, isdeepcopy=isdeepcopy)
                self.set_params(dom_cls, max_iter_samples, cat_end)
            elif isinstance(dataset, tc.Tensor):
                self.set_cls_list([dataset], isdeepcopy=isdeepcopy)
                self.set_params(dom_cls, max_iter_samples, cat_end)
            else:
                raise TypeError('dataset is a type error')

    @staticmethod
    def sum(sequence):
        r = 0
        L = len(sequence)
        for i in range(L):
            r += len(sequence[i])
        return r

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        L = len(sequence)
        for i in range(L):
            l = len(sequence[i])
            s += l
            r.append(s)
        return r

    @property
    def dom_cls(self):
        return self._dom_cls

    @dom_cls.setter
    def dom_cls(self, value):
        self.set_dom_cls(value)

    def set_params(self, dom_cls=None, max_iter_samples=0, cat_end=True):
        num_cls = len(self.cls_list)
        if num_cls > 0:
            if not bool(self.cls_labels) or (len(self.cls_labels) != num_cls):
                self.cls_labels = list(range(num_cls))
        else:
            self.cls_labels = []
        self.dom_cls = dom_cls
        self.set_sample_iter(max_iter_samples, cat_end)

    def set_dom_cls(self, dom_cls=None):
        if dom_cls is None:
            self._dom_cls = None
        else:
            num_cls = len(self.cls_list)
            if dom_cls < 0:
                self._dom_cls = num_cls + dom_cls
            else:
                self._dom_cls = dom_cls
            if self._dom_cls >= num_cls:
                self._dom_cls = num_cls - 1
            elif self._dom_cls < 0:
                self._dom_cls = 0
        return self._dom_cls

    def set_cls_labels(self, cls_labels=None):
        if cls_labels is None:
            self.cls_labels = []
        else:
            self.cls_labels = deepcopy(cls_labels)
        return self.cls_labels

    def set_cls_list(self, cls_list=None, cls_labels=None, isdeepcopy=False):
        assert isinstance(cls_list, list)
        if bool(cls_list):
            if isdeepcopy:
                 self.cls_list = deepcopy(cls_list)
            else:
                self.cls_list = copy(cls_list)
            num_cls = len(self.cls_list)
            if cls_labels is None or (len(cls_labels) != num_cls):
                self.cls_labels = list(range(num_cls))
            else:
                self.cls_labels = deepcopy(cls_labels)
        else:
            self.cls_list = []
            self.cls_labels = []
        return self.cls_list

    def set_dataset(self, dataset, num_cls=None):
        assert isinstance(dataset, data.Dataset), 'dataset must be a Dataset type'
        if num_cls is None:
            cls_labels = []
            cls_len = []
            for i in range(len(dataset)):
                ic = dataset[i][1]
                if tc.is_tensor(ic):
                    ic = int(ic.numpy())
                elif not isinstance(ic, str):
                    ic = int(ic)
                if ic in cls_labels:
                    ind = cls_labels.index(ic,0)
                    cls_len[ind] += 1
                else:
                    cls_labels.append(ic)
                    cls_len.append(1)
            num_cls = len(cls_labels)
        else:
            assert num_cls > 0, 'num_cls must be a positive integer'
            cls_labels = []
            cls_len = []
            for i in range(len(dataset)):
                ic = dataset[i][1]
                if tc.is_tensor(ic):
                    ic = int(ic.numpy())
                elif not isinstance(ic, str):
                    ic = int(ic)
                if 0 <= ic < num_cls:
                    if ic in cls_labels:
                        ind = cls_labels.index(ic,0)
                        cls_len[ind] += 1
                    else:
                        cls_labels.append(ic)
                        cls_len.append(1)
        if not isinstance(cls_labels[0], str):
            label = tc.Tensor(cls_labels)
            label = label.new_tensor(label, dtype=tc.int32)
            cls_labels, ind = tc.sort(label)
            cls_labels = cls_labels.tolist()
            cls_len = (tc.Tensor(cls_len)[ind]).tolist()
        shape = dataset[0][0].shape
        num_cls = len(cls_labels)
        cls_list = [0]*num_cls
        for m in range(num_cls):
            size = (cls_len[m],) + shape
            cls_list[m] = tc.zeros(size)
        cls_len = [0]*num_cls
        for i in range(len(dataset)):
            ic = dataset[i][1]
            if tc.is_tensor(ic):
                ic = int(ic.numpy())
            elif not isinstance(ic, str):
                ic = int(ic)
            if ic in cls_labels:
                ind = cls_labels.index(ic,0)
                n = cls_len[ind]
                cls_len[ind] = n + 1
                cls_list[ind][n] = dataset[i][0]
        self.cls_list = cls_list
        self.cls_labels = cls_labels
        return self.cls_list

    def set_sample_iter(self, max_iter_samples=None, cat_end=None):
        if max_iter_samples is None:
            max_iter_samples = self.max_iter_samples
        else:
            max_iter_samples = 0 if max_iter_samples < 0 else int(max_iter_samples)
            self.max_iter_samples = max_iter_samples
        if cat_end is None:
            cat_end = self.cat_end
        else:
            self.cat_end = bool(cat_end)
        if self.__len__() <= 0:
            self.sample_iter_list = []
            self.sample_iter = iter(self.sample_iter_list)
            self.max_iter_samples = max_iter_samples
            return self.sample_iter
        sample_iter = []
        if self.dom_cls is not None:
            N = len(self.cls_list[self.dom_cls])
            if max_iter_samples <= 0:
                max_iter_samples = N
            n0 = 0
            for n in range(0, N, max_iter_samples):
                n1 = n0 + max_iter_samples
                if n1 > N:
                    n1 = N
                    if cat_end and n > 0:
                        n0, _, _ = sample_iter[-1]
                        sample_iter.pop()
                sample_iter.append((n0, n1, self.dom_cls))
                n0 = n1
        else:
            if max_iter_samples <= 0:
                max_iter_samples = self.__len__()
            for i in range(len(self.cls_list)):
                N = len(self.cls_list[i])
                n0 = 0
                for n in range(0, N, max_iter_samples):
                    n1 = n0 + max_iter_samples
                    if n1 > N:
                        n1 = N
                        if cat_end and n > 0:
                            n0, _, _ = sample_iter[-1]
                            sample_iter.pop()
                    sample_iter.append((n0, n1, i))
                    n0 = n1
        self.sample_iter_list = sample_iter
        self.sample_iter = iter(self.sample_iter_list)
        return self.sample_iter

    def get_num_cls(self):
        return len(self.cls_list)

    def get_len_list(self):
        len_cls = []
        for cls in self.cls_list:
            len_cls.append(len(cls))
        return len_cls

    def get_len(self, ith_cls=0):
        return len(self.cls_list[ith_cls])

    def get_label(self, ith_cls=None):
        if ith_cls is None:
            return self.cls_labels
        return self.cls_labels[ith_cls]

    def get_cls(self, ith_cls=None):
        if ith_cls is None:
            return self.cls_list
        return self.cls_list[ith_cls]
    
    def get_index(self, label=None):
        if label is None:
            return list([i for i in range(len(self.cls_labels))])
        d = dict(zip(self.cls_labels, [i for i in range(len(self.cls_labels))]))
        if label in d.keys():
            return d[label]
        else:
            return None
        
    def get_cls_by_label(self, label=None):
        ind = list([i for i in range(len(self.cls_labels))])
        if label is None:
            return self.cls_list, ind
        d = dict(zip(self.cls_labels, ind))
        index = d[label]
        return self.cls_list[index], index

    def get_items_by_cls(self, ith_cls=0, index=None):
        if index is None:
            return self.cls_list[ith_cls], self.cls_labels[ith_cls]
        return self.cls_list[ith_cls][index], self.cls_labels[ith_cls]
    
    def get_item(self, index=0):
        if index is None:
            return None, None
        if self.dom_cls is not None:
            if index > len(self.cls_list[self.dom_cls])-1:
                return None, None
            return self.cls_list[self.dom_cls][index], self.cls_labels[self.dom_cls]
        if index > self.__len__()-1:
            return None, None
        if len(self.cls_list) == 1:
            return self.cls_list[0][index], self.cls_labels[0]
        LL0 = self.cumsum(self.cls_list)
        LL = [i-index-1 for i in LL0]#LL0 - index - 1
        if LL[0] >= 0:
            return self.cls_list[0][index], self.cls_labels[0]
        for i in range(1, len(LL)):
            if LL[i] >= 0:
                index = index - LL0[i-1]
                return self.cls_list[i][index], self.cls_labels[i]
        return None, None

    def data(self, ith_cls=None):
        vals = None
        labels = None
        L = self.get_num_cls()
        if ith_cls is None:
            ith_cls = range(L)
#        ipdb.set_trace()
        for i in ith_cls:
            if 0 <= i < L:
                di = self.cls_list[i]
                vals = di if vals is None else tc.cat((vals, di), 0)
                label = [self.cls_labels[i]]*di.size(0)
                labels = label if labels is None else labels + label
        return vals, labels

    def __getitem__(self, index=0):
        if index is None:
            return None, None
        if isinstance(index, int):
            return self.get_item(index)
        
        if isinstance(index, slice):
            L = self.__len__()
            stop = index.stop
            start = index.start
            step = index.step
            start = 0 if start is None else start
            step = 1 if step is None else step
            stop = L+stop if stop < 0 else stop
            stop = L if stop > L else stop
            index = list(range(start, stop, step))
        index = list(index)
        data = tc.Tensor()
        labels = []
        for i in index:
            d0, label = self.get_item(i)
            if d0 is not None:
                data = tc.cat((data, d0.unsqueeze(0)), 0)
                labels.append(label)
        if len(data)==0:
            return None, None
        return data, labels

    def __len__(self):
        if not bool(self.cls_list):
            return 0
        else:
            L = self.sum(self.cls_list)
            return L

    def __repr__(self):
        fmt_str = 'Dataset: ' + self.__class__.__name__ + '\n'
        fmt_str += '         Number of class: {}\n'.format(self.get_num_cls())
        fmt_str += '         Size of dataset: {}'.format(self.size())
        return fmt_str

    def __iter__(self):
        return self

    def __next__(self):
        n0, n1, i = next(self.sample_iter)
        return self.cls_list[i][n0:n1], self.cls_labels[i]

    def reiter(self, sample_iter_list=None):
        if sample_iter_list is not None:
            self.sample_iter_list = sample_iter_list
        self.sample_iter = iter(self.sample_iter_list)
        return self.sample_iter

    def leniter(self):
        return len(self.sample_iter_list)

    def size(self, dim=None):
        K = self.__len__()
        if K == 0:
            shape = (0, 0)
            shape = shape if dim is None else shape[dim]
            return shape
        else:
            shape = self.cls_list[0].size()
            shape = (K,) + shape[1:]
            shape = shape if dim is None else shape[dim]
            return shape

    def dim(self):
        K = self.__len__()
        if K == 0:
            return 0
        else:
            return self.cls_list[0].dim()

    def view(self, *args):
        obj = OrderedDataset()
        obj.copy(self)
        K = self.__len__()
        if K > 0:
            cls_list = []
            for i in range(len(self.cls_list)):
                if len(args) > 0:
                    if isinstance(args[0], tuple):
                        shape = (self.cls_list[i].shape[0],)
                        for s in args:
                            shape = shape + s
                    else:
                        shape = (self.cls_list[i].shape[0],) + args
                cls_list.append(self.cls_list[i].view(shape))
            obj.set_cls_list(cls_list, self.cls_labels)
        return obj

    def view_(self, *args):
        K = self.__len__()
        if K > 0:
            for i in range(len(self.cls_list)):
                if len(args) > 0:
                    if isinstance(args[0], tuple):
                        shape = (self.cls_list[i].shape[0],)
                        for s in args:
                            shape = shape + s
                    else:
                        shape = (self.cls_list[i].shape[0],) + args
                self.cls_list[i] = self.cls_list[i].view(shape)

        return self

    def copy(self, obj, isdeepcopy=False):
        assert isinstance(obj, OrderedDataset), 'obj must be an OrderedDataset type'
        if isdeepcopy:
            self.cls_list = deepcopy(obj.cls_list)
        else:
            self.cls_list = copy(obj.cls_list)
        self.cls_labels = deepcopy(obj.cls_labels)
        self.sample_iter_list = deepcopy(obj.sample_iter_list)
        self.sample_iter = deepcopy(obj.sample_iter)
        self.max_iter_samples = obj.max_iter_samples
        self.dom_cls = obj.dom_cls
        self.cat_end = obj.cat_end

    def clone(self, isdeepcopy=False):
        obj = OrderedDataset()
        obj.copy(self, isdeepcopy)
        return obj
    
    def apply(self, obj=None, fun=None, *args, **kwargs):
        if obj is None:
            F = getattr(self, fun, None)
        else:
            F = getattr(obj, fun, None)
        if F is None or len(self.cls_list)==0:
            return None
        cat_end = self.cat_end
        max_iter_samples = self.max_iter_samples
        output = []
        self.reiter()
#        ipdb.set_trace()
        for X, label in self:
            X = F(X, *args, **kwargs)
            m = self.get_index(label)
            M = len(output)
            if m >= M:
                output.append(X)
            else:
                output[m] = tc.cat((output[m], X), 0)
        output = OrderedDataset(output, max_iter_samples=max_iter_samples, cat_end=cat_end)
        output.cls_labels = deepcopy(self.cls_labels)
        self.reiter()
        return output


    def apply_(self, obj=None, fun=None, *args, **kwargs):
        if obj is None:
            F = getattr(self, fun, None)
        else:
            F = getattr(obj, fun, None)
        if F is None or len(self.cls_list)==0:
            return None
        self.reiter()
        output = []
        for X, label in self:
            X = F(X, *args, **kwargs)
            m = self.get_index(label)
            M = len(output)
            if m >= M:
                output.append(X)
                if m > 0:
                    self.cls_list[m-1] = None
            else:
                output[m] = tc.cat((output[m], X), 0)
        self.cls_list = output
        self.reiter()
        return self
    
    
class Ordered1DPatches(OrderedDataset):
    def __init__(self, dataset=None, kernel_size=3,
                 extraction_step=1, indent=0, max_patches=None,
                 seed=1, num_cls=None, dom_cls=None,
                 max_iter_samples=0, cat_end=True, isdeepcopy=False):
        super(Ordered1DPatches, self).__init__(dataset, num_cls, dom_cls,
                                         max_iter_samples, cat_end, isdeepcopy)
        self.kernel_size = kernel_size
        self.extraction_step = extraction_step
        self.indent = indent
        self.seed = seed
        self.num_patches = None
        if dataset is not None:
            self.extract_patches(kernel_size, extraction_step,
                                 indent, max_patches, seed)
            self.set_params(dom_cls, max_iter_samples, cat_end)

    def size(self, dim=None):
        K = self.__len__()
        if K == 0:
            shape = (0, 0, 0)
            shape = shape if dim is None else shape[dim]
            return shape
        else:
            shape = self.cls_list[0].size()
            shape = (K,) + shape[1:]
            shape = shape if dim is None else shape[dim]
            return shape

    def view(self, *args):
        obj = Ordered1DPatches()
        obj.copy(self)
        K = self.__len__()
        if K > 0:
            cls_list = []
            for i in range(len(self.cls_list)):
                if len(args) > 0:
                    if isinstance(args[0], tuple):
                        shape = (self.cls_list[i].shape[0],)
                        for s in args:
                            shape = shape + s
                    else:
                        shape = (self.cls_list[i].shape[0],) + args
                cls_list.append(self.cls_list[i].view(shape))
            obj.set_cls_list(cls_list, self.cls_labels)
        return obj

    def copy(self, obj, isdeepcopy=False):
        assert isinstance(obj, OrderedDataset), 'obj must be an OrderedDataset type'
        super(Ordered1DPatches, self).copy(obj, isdeepcopy)
        self.kernel_size = getattr(obj, 'kernel_size', 3)
        self.extraction_step = getattr(obj, 'extraction_step', 1)
        self.indent = getattr(obj, 'indent', 0)
        self.num_patches = getattr(obj, 'num_patches', None)
        self.seed = getattr(obj, 'seed', 1)

    def clone(self, isdeepcopy=False):
        obj = Ordered1DPatches()
        obj.copy(self, isdeepcopy)
        return obj

    def extract_patches(self, kernel_size=None, extraction_step=None,
                        indent=None, max_patches=None, seed=1):
        assert len(self.cls_list) > 0, 'self.cls_list must not be empty'
        if kernel_size is None:
            kernel_size = self.kernel_size
        if extraction_step is None:
            extraction_step = self.extraction_step
        if indent is None:
            indent = self.indent
        kernel_size, extraction_step, indent, num_patches = _parse_params_ext1d(
                    self.cls_list[0].shape, kernel_size, extraction_step, indent)
        if num_patches > 0 and kernel_size < self.cls_list[0].shape[2]:
            for i in range(len(self.cls_list)):
                self.cls_list[i] = extract_patches_1d(self.cls_list[i], kernel_size,
                                  extraction_step, indent, max_patches, seed)
        self.kernel_size = kernel_size
        self.extraction_step = extraction_step
        self.indent = indent
        self.num_patches = num_patches
        self.seed = seed


class Ordered2DPatches(OrderedDataset):
    def __init__(self, dataset=None, kernel_size=3,
                 extraction_step=1, indent=0, max_patches=None,
                 seed=1, num_cls=None, dom_cls=None,
                 max_iter_samples=0, cat_end=True, isdeepcopy=False):
        super(Ordered2DPatches, self).__init__(dataset, num_cls, dom_cls,
                                         max_iter_samples, cat_end, isdeepcopy)
        self.kernel_size = kernel_size
        self.extraction_step = extraction_step
        self.indent = indent
        self.seed = seed
        self.num_patches = None
        if dataset is not None:
#            ipdb.set_trace()
            self.extract_patches(kernel_size, extraction_step,
                                 indent, max_patches, seed)
            self.set_params(dom_cls, max_iter_samples, cat_end)

    def size(self, dim=None):
        K = self.__len__()
        if K == 0:
            shape = (0, 0, 0, 0)
            shape = shape if dim is None else shape[dim]
            return shape
        else:
            shape = self.cls_list[0].size()
            shape = (K,) + shape[1:]
            shape = shape if dim is None else shape[dim]
            return shape

    def view(self, *args):
        obj = Ordered2DPatches()
        obj.copy(self)
        K = self.__len__()
        if K > 0:
            cls_list = []
            for i in range(len(self.cls_list)):
                if len(args) > 0:
                    if isinstance(args[0], tuple):
                        shape = (self.cls_list[i].shape[0],)
                        for s in args:
                            shape = shape + s
                    else:
                        shape = (self.cls_list[i].shape[0],) + args
                cls_list.append(self.cls_list[i].view(shape))
            obj.set_cls_list(cls_list, self.cls_labels)
        return obj

    def copy(self, obj, isdeepcopy=False):
        assert isinstance(obj, OrderedDataset), 'obj must be an OrderedDataset type'
        super(Ordered2DPatches, self).copy(obj, isdeepcopy)
        self.kernel_size = getattr(obj, 'kernel_size', 3)
        self.extraction_step = getattr(obj, 'extraction_step', 1)
        self.indent = getattr(obj, 'indent', 0)
        self.num_patches = getattr(obj, 'num_patches', None)
        self.seed = getattr(obj, 'seed', 1)

    def clone(self, isdeepcopy=False):
        obj = Ordered2DPatches()
        obj.copy(self, isdeepcopy)
        return obj

    def extract_patches(self, kernel_size=None, extraction_step=None,
                        indent=None, max_patches=None, seed=1):
        assert len(self.cls_list) > 0, 'self.cls_list must not be empty'
        if kernel_size is None:
            kernel_size = self.kernel_size
        if extraction_step is None:
            extraction_step = self.extraction_step
        if indent is None:
            indent = self.indent
        kernel_size, extraction_step, indent, num_patches = _parse_params_ext2d(
                    self.cls_list[0].shape, kernel_size, extraction_step, indent)
#        ipdb.set_trace()
        if num_patches > 0 and kernel_size != self.cls_list[0].shape[2:4]:
            for i in range(len(self.cls_list)):
                self.cls_list[i] = extract_patches_2d(self.cls_list[i],
                     kernel_size, extraction_step, indent, max_patches, seed)
        self.kernel_size = kernel_size
        self.extraction_step = extraction_step
        self.indent = indent
        self.num_patches = num_patches
        self.seed = seed


class Ordered3DPatches(OrderedDataset):
    def __init__(self, dataset=None, kernel_size=3,
                 extraction_step=1, indent=0, max_patches=None,
                 seed=1, num_cls=None, dom_cls=None,
                 max_iter_samples=0, cat_end=True, isdeepcopy=False):
        super(Ordered3DPatches, self).__init__(dataset, num_cls, dom_cls,
                                         max_iter_samples, cat_end, isdeepcopy)
        self.kernel_size = kernel_size
        self.extraction_step = extraction_step
        self.indent = indent
        self.seed = seed
        self.num_patches = None
        if dataset is not None:
#            ipdb.set_trace()
            self.extract_patches(kernel_size, extraction_step,
                                 indent, max_patches, seed)
            self.set_params(dom_cls, max_iter_samples, cat_end)

    def size(self, dim=None):
        K = self.__len__()
        if K == 0:
            shape = (0, 0, 0, 0, 0)
            shape = shape if dim is None else shape[dim]
            return shape
        else:
            shape = self.cls_list[0].size()
            shape = (K,) + shape[1:]
            shape = shape if dim is None else shape[dim]
            return shape

    def view(self, *args):
        obj = Ordered3DPatches()
        obj.copy(self)
        K = self.__len__()
        if K > 0:
            cls_list = []
            for i in range(len(self.cls_list)):
                if len(args) > 0:
                    if isinstance(args[0], tuple):
                        shape = (self.cls_list[i].shape[0],)
                        for s in args:
                            shape = shape + s
                    else:
                        shape = (self.cls_list[i].shape[0],) + args
                cls_list.append(self.cls_list[i].view(shape))
            obj.set_cls_list(cls_list, self.cls_labels)
        return obj

    def copy(self, obj, isdeepcopy=False):
        assert isinstance(obj, OrderedDataset), 'obj must be an OrderedDataset type'
        super(Ordered3DPatches, self).copy(obj, isdeepcopy)
        self.kernel_size = getattr(obj, 'kernel_size', 3)
        self.extraction_step = getattr(obj, 'extraction_step', 1)
        self.indent = getattr(obj, 'indent', 0)
        self.num_patches = getattr(obj, 'num_patches', None)
        self.seed = getattr(obj, 'seed', 1)

    def clone(self, isdeepcopy=False):
        obj = Ordered3DPatches()
        obj.copy(self, isdeepcopy)
        return obj

    def extract_patches(self, kernel_size=None, extraction_step=None,
                        indent=None, max_patches=None, seed=1):
        assert len(self.cls_list) > 0, 'self.cls_list must not be empty'
        if kernel_size is None:
            kernel_size = self.kernel_size
        if extraction_step is None:
            extraction_step = self.extraction_step
        if indent is None:
            indent = self.indent
        kernel_size, extraction_step, indent, num_patches = _parse_params_ext3d(
                    self.cls_list[0].shape, kernel_size, extraction_step, indent)
#        ipdb.set_trace()
        if num_patches > 0 and kernel_size != self.cls_list[0].shape[2:5]:
            for i in range(len(self.cls_list)):
                self.cls_list[i] = extract_patches_3d(self.cls_list[i],
                     kernel_size, extraction_step, indent, max_patches, seed)
        self.kernel_size = kernel_size
        self.extraction_step = extraction_step
        self.indent = indent
        self.num_patches = num_patches
        self.seed = seed
        
#%%      
def extract_patches_1d(sequs, kernel_size=3, extraction_step=1,
                       indent=0, max_patches=None, seed=1):
    sequs_shape = sequs.shape
    kernel_size, extraction_step, indent, num_patches = _parse_params_ext1d(
                               sequs_shape, kernel_size, extraction_step, indent)
    i_b, i_c, i_h = sequs_shape
    i_h0 = indent
    i_h1 = i_h - kernel_size - indent + 1
    shape = (i_b*num_patches, i_c) + (kernel_size,)
    outputs = tc.zeros(shape)
    m = 0
    for i in tc.arange(i_h0, i_h1, extraction_step, dtype=tc.int):
        outputs[m:m+i_b] = sequs[:,:,i:(i+kernel_size)]
        m = m + i_b
    if max_patches is not None and (1<=max_patches<shape[0]):
        tc.manual_seed(seed)
        ind = tc.randperm(shape[0])[0:int(max_patches)].tolist()
        outputs = outputs[ind]
    return outputs


def extract_patches_2d(images, kernel_size=3, extraction_step=1,
                       indent=0, max_patches=None, seed=1):
    imgs_shape = images.shape
    kernel_size, extraction_step, indent, num_patches = _parse_params_ext2d(
                               imgs_shape, kernel_size, extraction_step, indent)
    d_h, d_w = indent
    p_h, p_w = kernel_size
    i_b, i_c, i_h, i_w = imgs_shape
    i_h0 = d_h
    i_h1 = i_h - p_h - d_h + 1
    i_w0 = d_w
    i_w1 = i_w - p_w - d_w + 1
    shape = (i_b*num_patches, i_c) + kernel_size
    outputs = tc.zeros(shape)
    m = 0
    s_h, s_w = extraction_step
#    ipdb.set_trace()
    for i in tc.arange(i_h0, i_h1, s_h, dtype=tc.int):
        for j in tc.arange(i_w0, i_w1, s_w, dtype=tc.int):
            outputs[m:m+i_b] = images[:,:,i:(i+p_h),j:(j+p_w)]
            m = m + i_b
    if max_patches is not None and (1<=max_patches<shape[0]):
        tc.manual_seed(seed)
        ind = tc.randperm(shape[0])[0:int(max_patches)].tolist()
        outputs = outputs[ind]
    return outputs


def extract_patches_3d(images, kernel_size=3, extraction_step=1,
                       indent=0, max_patches=None, seed=1):
    imgs_shape = images.shape
    kernel_size, extraction_step, indent, num_patches = _parse_params_ext3d(
                               imgs_shape, kernel_size, extraction_step, indent)
    d_d, d_h, d_w = indent
    p_d, p_h, p_w = kernel_size
    i_b, i_c, i_d, i_h, i_w = imgs_shape
    i_d0 = d_d
    i_d1 = i_d - p_d - d_d + 1
    i_h0 = d_h
    i_h1 = i_h - p_h - d_h + 1
    i_w0 = d_w
    i_w1 = i_w - p_w - d_w + 1
    shape = (i_b*num_patches, i_c) + kernel_size
    outputs = tc.zeros(shape)
    m = 0
    s_d, s_h, s_w = extraction_step
#    ipdb.set_trace()
    for k in tc.arange(i_d0, i_d1, s_d, dtype=tc.int):
        for i in tc.arange(i_h0, i_h1, s_h, dtype=tc.int):
            for j in tc.arange(i_w0, i_w1, s_w, dtype=tc.int):
                outputs[m:m+i_b] = images[:,:,k:(k+p_d),k:(k+p_d),j:(j+p_w)]
                m = m + i_b
    if max_patches is not None and (1<=max_patches<shape[0]):
        tc.manual_seed(seed)
        ind = tc.randperm(shape[0])[0:int(max_patches)].tolist()
        outputs = outputs[ind]
    return outputs


def _parse_params_ext1d(sequs_shape, kernel_size=3, extraction_step=1, indent=0):
    i_b, i_c, i_h = sequs_shape
    if indent is None:
        indent = 0
    elif isinstance(indent, Iterable):
        indent = int(indent[0])
    assert indent >= 0, 'indent < 0'
    if extraction_step is None:
        extraction_step = 1
    elif isinstance(extraction_step, Iterable):
        extraction_step = int(extraction_step[0])
    assert extraction_step >= 1, 'extraction_step < 1'
    if kernel_size is None:
        kernel_size = i_h - 2*indent
    elif isinstance(kernel_size, Iterable):
        kernel_size = int(kernel_size[0])
    assert kernel_size > 1, 'kernel_size <= 1'
    i_h0 = indent
    i_h1 = i_h - kernel_size - indent + 1
    num_patches = int(len(tc.arange(i_h0, i_h1, extraction_step)))
    return kernel_size, extraction_step, indent, num_patches


def _parse_params_ext2d(imgs_shape, kernel_size=3, extraction_step=1, indent=0):
    i_b, i_c, i_h, i_w = imgs_shape
    if indent is None:
        indent = (0, 0)
    elif isinstance(indent, Iterable):
        indent = (int(indent[0]), int(indent[1]))
    if not isinstance(indent, tuple):
        assert indent >= 0, 'indent < 0'
        indent = (indent, indent)
    if extraction_step is None:
        extraction_step = (1, 1)
    elif isinstance(extraction_step, Iterable):
        extraction_step = (int(extraction_step[0]), int(extraction_step[1]))
    if not isinstance(extraction_step, tuple):
        assert extraction_step >= 1, 'extraction_step < 1'
        extraction_step = (extraction_step, extraction_step)
    d_h, d_w = indent
    if kernel_size is None:
        assert i_h-2*d_h > 0 and 0 < i_w-2*d_w, 'i_h-2*d_h <= 0 or i_w-2*d_w <= 0'
        kernel_size = (i_h-2*d_h, i_w-2*d_w)
    elif isinstance(kernel_size, Iterable):
        kernel_size = (kernel_size[0], kernel_size[1])
    if not isinstance(kernel_size, tuple):
        assert kernel_size > 1, 'kernel_size <= 1'
        kernel_size = (kernel_size, kernel_size)
    p_h, p_w = kernel_size
    i_h0 = d_h
    i_h1 = i_h - p_h - d_h + 1
    i_w0 = d_w
    i_w1 = i_w - p_w - d_w + 1
    s_h, s_w = extraction_step
    num_patches = int(len(tc.arange(i_h0, i_h1, s_h))
                      *len(tc.arange(i_w0, i_w1, s_w)))
    return kernel_size, extraction_step, indent, num_patches


def _parse_params_ext3d(imgs_shape, kernel_size=3, extraction_step=1, indent=0):
    i_b, i_c, i_d, i_h, i_w = imgs_shape
    if indent is None:
        indent = (0, 0, 0)
    elif isinstance(indent, Iterable):
        indent = (int(indent[0]), int(indent[1]), int(indent[2]))
    if not isinstance(indent, tuple):
        assert indent >= 0, 'indent < 0'
        indent = (indent, indent, indent)
    if extraction_step is None:
        extraction_step = (1, 1, 1)
    elif isinstance(extraction_step, Iterable):
        extraction_step = (int(extraction_step[0]), int(extraction_step[1]), int(extraction_step[2]))
    if not isinstance(extraction_step, tuple):
        assert extraction_step >= 1, 'extraction_step < 1'
        extraction_step = (extraction_step, extraction_step, extraction_step)
    d_d, d_h, d_w = indent
    if kernel_size is None:
        assert i_h-2*d_h > 0 and 0 < i_w-2*d_w, 'i_h-2*d_h <= 0 or i_w-2*d_w <= 0'
        kernel_size = (i_d-2*d_d, i_h-2*d_h, i_w-2*d_w)
    elif isinstance(kernel_size, Iterable):
        kernel_size = (kernel_size[0], kernel_size[1], kernel_size[2])
    if not isinstance(kernel_size, tuple):
        assert kernel_size > 1, 'kernel_size <= 1'
        kernel_size = (kernel_size, kernel_size, kernel_size)
    p_d, p_h, p_w = kernel_size
    i_d0 = d_d
    i_d1 = i_d - p_d - d_d + 1
    i_h0 = d_h
    i_h1 = i_h - p_h - d_h + 1
    i_w0 = d_w
    i_w1 = i_w - p_w - d_w + 1
    s_d, s_h, s_w = extraction_step
    num_patches = int(len(tc.arange(i_d0, i_d1, s_d))*len(tc.arange(i_h0, i_h1, s_h))
                      *len(tc.arange(i_w0, i_w1, s_w)))
    return kernel_size, extraction_step, indent, num_patches
