from abc import abstractmethod

import torch

from tensorCache.abstractTorchCache import AbstractTorchCache


class TensorListCache(AbstractTorchCache):

    def __init__(self, cache_size, max_index, tensor_shapes, dtypes, data_device, logic_device="cpu"):
        super().__init__(cache_size, max_index, data_device, logic_device)
        self.tensor_shapes = tensor_shapes
        self.dtypes = dtypes
        # Initialize multiple caches
        self.caches = [torch.empty((self.cache_size, *shape), dtype=dtype, device=self.data_device)
                       for shape, dtype in zip(self.tensor_shapes, self.dtypes)]

    @abstractmethod
    def compute_idxs(self, idxs):
        raise NotImplementedError("Implement this method to use the TensorListCache object")


    def _insert_data(self, tensor_data_list, lru_slots):
        for cache, tensor_data in zip(self.caches, tensor_data_list):
            cache[lru_slots] = tensor_data


    def _get_empty_output(self, n):
        return [torch.empty((n,) + shape, dtype=dtype, device=self.data_device)
                for shape, dtype in zip(self.tensor_shapes, self.dtypes)]


    def _set_output_values(self, output_tensors, output_idxs, cache_idxs):
        for output_tensor, cache in zip(output_tensors, self.caches):
            if len(cache_idxs) > 0:
                output_tensor[output_idxs] = cache[cache_idxs]