# Torch cache
This implements an LRU cache for tensors.
You just need to define a child class inheriting from `TensorCache` and implement the
compute_idxs method, a method that compute the values of the cache for the given indices
```

from tensorCache.tensorCache import TensorCache

class TestTensorCache(TensorCache):
    def compute_idxs(self, idxs):
        values = torch.ones(len(idxs)) * torch.as_tensor(idxs)
        return torch.ones((len(idxs),) + self.tensor_shape) * values.unsqueeze(-1).unsqueeze(-1)
```