from abc import abstractmethod

import torch

from tensorCache.abstractTorchCache import AbstractTorchCache


class TensorCache(AbstractTorchCache):
    def __init__(self, cache_size, tensor_shape, max_index, dtype, device):
        super().__init__(cache_size, max_index, device)
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.cache = torch.zeros((self.cache_size, *self.tensor_shape), dtype=dtype, device=device)

    @abstractmethod
    def compute_idxs(self, idxs):
        raise NotImplementedError("Implement this method to use the torchCache object ")

    def _insert_data(self, tensor_data, lru_slots):
        self.cache[lru_slots] = tensor_data

    def _get_empty_output(self, n):
        return torch.empty((n,)+self.tensor_shape, dtype=self.dtype, device=self.device)

    def _set_output_values(self, output_tensor, output_idxs, cache_idxs):
        if len(cache_idxs) > 0:
            output_tensor[output_idxs] = self.cache[cache_idxs]