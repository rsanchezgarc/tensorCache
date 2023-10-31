from abc import ABC, abstractmethod
import torch

class AbstractTorchCache(ABC):
    def __init__(self, cache_size, max_index, device):
        self.cache_size = cache_size
        self.max_index = max_index
        self.device = device

        self.index_tracker = torch.full((self.max_index,), -1, dtype=torch.long, device=device)
        self.reverse_index_tracker = torch.full((self.cache_size,), -1, dtype=torch.long, device=device)
        self.lru_tracker = torch.zeros(self.cache_size, dtype=torch.long, device=device)
        self.current_lru = 0

    @abstractmethod
    def compute_idxs(self, idxs):
        raise NotImplementedError("This method needs to be defined by the user to compute tensor(s) values given idxs")

    @abstractmethod
    def _insert_data(self, tensor_data, lru_slots):
        #self.cache[lru_slots] = tensor_data
        raise NotImplementedError()

    @abstractmethod
    def _get_empty_output(self, n):
        # return torch.empty(n, dtype=self.dtype, device=self.device)
        raise NotImplementedError()

    @abstractmethod
    def _set_output_values(self, output_tensor, output_idxs, cache_idxs):
        # return output_tensor[output_idxs] = self.cache[cache_idxs]
        raise NotImplementedError()


    def insert(self, index_tensors, tensor_data):

        k = len(index_tensors)
        assert k <= len(self.lru_tracker), f"Error, too many indices {k} to be inserted in a cache of size {len(self.lru_tracker)}"

        _, lru_slots = torch.topk(self.lru_tracker, k, largest=False)

        old_indices = self.reverse_index_tracker[lru_slots]
        valid_old_indices = old_indices != -1
        self.index_tracker[old_indices[valid_old_indices]] = -1

        self.reverse_index_tracker[lru_slots] = index_tensors

        self._insert_data(tensor_data, lru_slots)
        self.index_tracker[index_tensors] = lru_slots

        self.lru_tracker[lru_slots] = self.current_lru + 1
        self.current_lru += 1
        #TODO: how to deal with overflow

    def retrieve(self, index_tensors):

        output_tensor = self._get_empty_output(len(index_tensors))

        cache_slots = self.index_tracker[index_tensors]
        valid_slots = cache_slots != -1
        self._set_output_values(output_tensor, valid_slots, cache_slots[valid_slots])

        missing_indices = index_tensors[~valid_slots]
        if len(missing_indices) > 0:
            unique_missing_indices = torch.unique(missing_indices)
            unique_computed_data = self.compute_idxs(unique_missing_indices)
            self.insert(unique_missing_indices, unique_computed_data)
            # They are now, for sure, in the cache
            # output_tensor[~valid_slots] = self.cache[self.index_tracker[missing_indices]]
            self._set_output_values(output_tensor, ~valid_slots, self.index_tracker[missing_indices])

        self.lru_tracker[cache_slots[valid_slots]] = self.current_lru + 1
        self.current_lru += 1

        return output_tensor

    def __getitem__(self, item):
        return self.retrieve(item)
