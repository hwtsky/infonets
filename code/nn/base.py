# base.py
"""
Created on Sun May  6 12:44:12 2018

@author: Wentao Huang
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict, Iterable
from itertools import islice
import operator
import torch as tc
from ..utils.helper import is_number, get_key_val
from ..utils.data import OrderedDataset

__all__ = [
    "Base",
    "Sequential",
    "ModuleList",
    "ParameterList",
]

class Base(tc.nn.Module):
    
    def __init__(self):
        self.reinit()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self.set_name(value)

    def set_name(self, name=None):
        if name is None:
            name = 'None'
        elif is_number(name, is_raised=False):
            name = str(name)
        if not isinstance(name, str):
            raise TypeError("module name is not a string")
        if '.' in name:
            raise ValueError("module name can't contain \".\"")
        elif name == '':
            raise ValueError("module name can't be empty string \"\"")
        self._name = name
        return name

    @property
    def learning_method(self):
        return self._learning_method

    @learning_method.setter
    def learning_method(self, value):
        self._learning_method = value

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self.set_input(value)

    def reinit(self):
        super(Base, self).__init__()
        self._name = 'None'
        self._input = None # an OrderedDataset type
        self._learning_method = None
        self._parameters_train = OrderedDict()
    
    def empty_parameters(self):
        for key in self._parameters.keys():
            self._parameters[key] = None
        
    def empty_buffers(self):
        for key in self._buffers.keys():
            self._buffers[key] = None
            
    def set_input(self, input=None):
        if input is None:
            self._input = None
        elif isinstance(input, OrderedDataset):
            self._input = input
        else:
            self._input = OrderedDataset(input)
        return self._input

    def set_parameters_train(self, params=None, **kwargs):
        if params is not None:
            self._parameters_train.update(params)
        if bool(kwargs):
            self._parameters_train.update(kwargs)
        return self._parameters_train

    def get_parameters_train(self, key=None, assign=None, is_raised=False,
                                 err='get_parameters_train'):
        if key is None:
            return self._parameters_train
        return get_key_val(key=key, assign=assign, is_raised=is_raised,
                err=err, params=self._parameters_train)

    def get_parameters(self, key=None, assign=None, is_raised=False,
                           err='get_parameters'):
        if key is None:
            return self._parameters
        return get_key_val(key=key, assign=assign, is_raised=is_raised,
                err=err, params=self._parameters)

    def register_buffer(self, name, tensor):
        r"""Adds a persistent buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the persistent state.

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor): buffer to be registered.

        Example::

            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif (tensor is not None and #not isinstance(tensor, list) and 
             not isinstance(tensor, tc.Tensor)):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch Tensor or None required)"
                            .format(tc.typename(tensor), name))
        else:
            self._buffers[name] = tensor
            
    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            parameter (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("module name is not a string")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, list) and not isinstance(param, tc.nn.Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(Parameter or None required)"
                            .format(tc.typename(param), name))
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param

    def parameters_children(self):
        r"""Returns an iterator immediate module parameters.

        This is typically passed to an optimizer.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters_children():
            yield param

    def named_parameters_children(self, memo=None, prefix=''):
        r"""Returns an iterator immediate module parameters, yielding both the
        name of the parameter as well as the parameter itself

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        if memo is None:
            memo = set()
        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
                
    def add_module(self, name=None, module=None):
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            parameter (Module): child module to be added to the module.
        """
        if name is None:
            name = 'None'
        elif is_number(name, is_raised=False):
            name = str(name)
        if not isinstance(name, str):
            raise TypeError("module name is not a string")
        if module is None:
            raise ValueError("module is None")
        elif not isinstance(module, tc.nn.Module):
            raise TypeError("{} is not a Module subclass".format(
                tc.typename(module)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        module.name = name
        self._modules[module.name] = module

    def train_exe(self, input=None, params=None, **kwargs):
        r"""Execute training

        Should be overridden by subclasses.
        """
        if self.training and (self.learning_method is not None):
            input = self.input if input is None else input
            if input is None:
                raise ValueError("{}.train_exe(input): input is None".format(self.name))
            else:
                return NotImplementedError
        else:
            return None


class Sequential(Base):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def train_exe(self, input=None, params=None, **kwargs):
        flag = None
        for module in self._modules.values():
            flag0 = module.train_exe(input, params=params, **kwargs)
            input = module(input)
            if flag0 is not None:
                flag = True
        return flag



class ModuleList(Base):
    r"""Holds submodules in a list.

    ModuleList can be indexed like a regular Python list, but modules it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = operator.index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, module):
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self



class ParameterList(Base):
    r"""Holds parameters in a list.

    ParameterList can be indexed like a regular Python list, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        parameters (iterable, optional): an iterable of :class:`~torch.nn.Parameter`` to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        if parameters is not None:
            self += parameters

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ParameterList(list(self._parameters.values())[idx])
        else:
            idx = operator.index(idx)
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range'.format(idx))
            if idx < 0:
                idx += len(self)
            return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        idx = operator.index(idx)
        return self.register_parameter(str(idx), param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, parameter):
        """Appends a given parameter at the end of the list.

        Arguments:
            parameter (nn.Parameter): parameter to append
        """
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters):
        """Appends parameters from a Python iterable to the end of the list.

        Arguments:
            parameters (iterable): iterable of parameters to append
        """
        if not isinstance(parameters, Iterable):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self

    def extra_repr(self):
        tmpstr = ''
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                tc.typename(p.data), size_str, device_str)
            tmpstr = tmpstr + '  (' + k + '): ' + parastr + '\n'
        return tmpstr

