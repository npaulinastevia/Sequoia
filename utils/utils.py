""" Set of Utilities. """

import collections
from collections import defaultdict, deque
from collections.abc import MutableMapping
from typing import (Any, Deque, Dict, Iterable, List, MutableMapping, Optional,
                    Tuple, TypeVar, Union)

import torch
from torch import Tensor, nn

cuda_available = torch.cuda.is_available()
gpus_available = torch.cuda.device_count()

T = TypeVar("T")

def n_consecutive(items: Iterable[T], n: int=2, yield_last_batch=True) -> Iterable[Tuple[T, ...]]:
    values: List[T] = []
    for item in items:
        values.append(item)
        if len(values) == n:
            yield tuple(values)
            values.clear()
    if values and yield_last_batch:
        yield tuple(values)


def to_list(tensors: Iterable[Union[T, Tensor]]) -> List[T]:
    """Converts a list of tensors into a list of values.
    
    `tensots` must contain scalar tensors.Any
    
    Parameters
    ----------
    - tensors : Iterable[Tensor]
    
        some scalar tensors
    
    Returns
    -------
    List[float]
        A list of their values.
    """
    if tensors is None:
        return []
    return list(map(lambda v: v.item() if isinstance(v, Tensor) else v, tensors))


def to_dict_of_lists(list_of_dicts: List[Dict[str, Tensor]]) -> Dict[str, List[Tensor]]:
    # TODO: we have a list of dicts, change it into a dict of lists.
    result: Dict[str, List[Any]] = defaultdict(list)
    for i, d in enumerate(list_of_dicts):
        for key, tensor in d.items():
            result[key].append(tensor.cpu())
        assert d.keys() == result.keys()
    return result


def add_prefix(some_dict: Dict[str, T], prefix: str="") -> Dict[str, T]:
    return {prefix + key: value for key, value in some_dict.items()}


def loss_str(loss_tensor: Tensor) -> str:
    loss = loss_tensor.item()
    if loss == 0:
        return "0"
    elif abs(loss) < 1e-3 or abs(loss) > 1e3:
        return f"{loss:.1e}"
    else:
        return f"{loss:.3f}"

import functools

def rsetattr(obj: Any, attr: str, val: Any) -> None:
    """ Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj: Any, attr: str, *args):
    """ Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class TensorCache(MutableMapping[Tensor, Tensor]):
    """A mutable mapping of individual (not batched) tensors to their outputs.

    TODO: Not really useful, since the weights keep changing anyway. Might get rid of this entirely.
    """
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.x_shape: Optional[torch.Size] = None
        self.y_shape: Optional[torch.Size] = None
        self.x_cache: Optional[Tensor] = None
        self.y_cache: Optional[Tensor] = None
        self.head: int = 0
        self.tail: int = 0

    def __contains__(self, item: Any) -> bool:
        # TODO: vectorize this.
        if isinstance(item, Tensor):     
            for x, y in self:
                if (x == item).all():
                    return True
        return False
        
    def __getitem__(self, key: Tensor) -> Tensor:
        if self.y_cache is None:
            self.x_shape = key.shape
            raise KeyError(key)

        for x, y in self:
            if (x == item).all():
                return y
        # shouldn't happen, cause __contains__ is called before __getitem__ ?
        raise KeyError(key) 

    def __setitem__(self, key: Tensor, value: Tensor) -> None:
        if self.x_cache is None:
            self.x_shape = key.shape
            self.x_cache = key.new_zeros([self.capacity, *self.x_shape])
        if self.y_cache is None:
            self.y_shape = value.shape
            self.y_cache = value.new_zeros([self.capacity, *self.y_shape])
        self.head += 1
        self.head %= self.capacity
        self.x_cache[self.head] = key
        self.y_cache[self.head] = value
    
    def __len__(self) -> int:
        return ((self.head + self.capacity) - self.tail) % self.capacity

    def __delitem__(self, key: Tensor) -> None:
        pass

    def __iter__(self):
        return iter(zip(self.x_cache, self.y_cache))
    
    def clear(self) -> None:
        del self.x_cache
        del self.y_cache
        self.x_cache = None
        self.y_cache = None


class CachedForwardPass(nn.Module):
    """TODO: create a wrapper that can cache results of a forward pass
    Thoughts:
    - Makes no sense, since whenever the model updates you will clear the cache anyway!
    - Use a tensor to hold the results? or use deque? 
    - Check membership at an example level?
    - How should this be used? as a decorator on the forward method? or as a base class?
    """
    def __init__(self, capacity: int = 100):
        super().__init__()
        self.capacity = capacity
        self.cache = TensorCache(capacity=capacity)
        self.head: int = 0
        self.tail: int = 0

    def forward(self, x_batch: Tensor):
        x_s = x_batch
        is_cached = [x in self for x in x_batch]
        print(is_cached)
        return False
            

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # cache = TensorCache(5)


    # d = TensorCache(5)
    # zero = torch.zeros(3,3)
    # d[zero] = torch.Tensor(123)

    # one = torch.ones(3,3)
    # batch = torch.stack([zero, one])
    # print("zero is in cache:", zero in d)
    # print("ones is in cache:", one in d)
    # print(torch.zeros(3,3) in d)
    # print(d[torch.zeros(3,3)])
