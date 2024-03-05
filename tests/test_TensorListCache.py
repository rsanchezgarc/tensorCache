import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from tensorCache.tensorListLRUCache import TensorListLRUCache


DTYPES = [torch.float32, torch.float64]
SHAPES = [(2, 2), (3, 3)]
class TensorListCacheForTest(TensorListLRUCache):
    def compute_idxs(self, idxs):
        # Assume a simple computation for mock
        return [torch.ones((len(idxs),) + self.caches[i].shape[1:], dtype=DTYPES[i]) * i
                for i in range(len(self.caches))]


# Test class for TensorListLRUCache
class TestTensorListCache(unittest.TestCase):
    def setUp(self):
        self.cache_size = 5
        self.max_index = 20
        self.tensor_shapes = [(2, 2), (3, 3)]
        self.dtypes = DTYPES
        self.device = "cuda" #"cpu"
        self.cache = TensorListCacheForTest(self.cache_size, self.max_index, self.tensor_shapes, self.dtypes, self.device)

    def test_arbitrary_indexing(self):
        num_iterations = 200
        cum_cache = [0] * len(self.tensor_shapes)
        cum_direct = [0] * len(self.tensor_shapes)

        for i in range(num_iterations):
            idx_subset = torch.randint(0, self.max_index, (np.random.randint(1, self.cache_size),))

            direct_values = self.cache.compute_idxs(idx_subset)
            cache_values = self.cache.retrieve(idx_subset)

            for j in range(len(self.tensor_shapes)):
                cum_direct[j] += direct_values[j].sum()
                cum_cache[j] += cache_values[j].sum()

        for i in range(len(self.tensor_shapes)):
            self.assertAlmostEqual(cum_direct[i].item(), cum_cache[i].item(), places=2)



    def test_single_insert_and_retrieve(self):

        idx = torch.tensor([1])
        expected_data = self.cache._compute_idxs(idx)
        self.cache.insert(idx, expected_data)

        retrieved_data = self.cache.retrieve(idx)
        for i in range(len(self.tensor_shapes)):
            self.assertTrue(torch.equal(retrieved_data[i], expected_data[i]))

    def test_multiple_insert_and_retrieve(self):

        idx = torch.tensor([1, 2, 3])
        expected_data = self.cache._compute_idxs(idx)
        self.cache.insert(idx, expected_data)
        retrieved_data = self.cache.retrieve(idx)
        for i in range(len(self.tensor_shapes)):
            self.assertTrue(torch.equal(retrieved_data[i], expected_data[i]))



    def test_lru_eviction_single(self):

        idx = torch.tensor([1, 2, 3, 4, 5])
        expected_data = self.cache._compute_idxs(idx)
        self.cache.insert(idx, expected_data)

        new_idx = torch.tensor([6])
        self.cache.retrieve(new_idx)


        # The least recently used item (1) should be evicted, so it should be recomputed
        final_retrieval_idx = torch.tensor([1, 6])
        final_retrieval_data = self.cache.retrieve(final_retrieval_idx)

        final_expected_data = self.cache._compute_idxs(final_retrieval_idx)
        for i in range(len(self.tensor_shapes)):
            self.assertTrue(torch.equal(final_retrieval_data[i], final_expected_data[i]))


    def test_lru_eviction_multiple(self):

        idx = torch.tensor([1, 2, 3, 4, 5])
        expected_data = self.cache._compute_idxs(idx)
        self.cache.insert(idx, expected_data)

        new_idx = torch.tensor([6, 7])
        self.cache.retrieve(new_idx)

        # The least recently used items (1, 2) should be evicted, so they should be recomputed

        final_retrieval_idx = torch.tensor([1, 2, 6, 7])
        final_retrieval_data = self.cache.retrieve(final_retrieval_idx)

        final_expected_data = self.cache._compute_idxs(final_retrieval_idx)
        for i in range(len(self.tensor_shapes)):
            self.assertTrue(torch.equal(final_retrieval_data[i], final_expected_data[i]))


    def test_sequential_traversal(self):

        oridata = torch.arange(self.max_index)
        dataset = TensorDataset(oridata)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        avg_list = [[] for _ in self.tensor_shapes]
        for i, data in enumerate(dataloader):

            idx = data[0]
            tensor_list = self.cache.retrieve(idx)
            for j in range(len(self.tensor_shapes)):
                avg = torch.mean(tensor_list[j])
                avg_list[j].append(avg.item())

        for i in range(len(self.tensor_shapes)):

            avg_result = np.mean(avg_list[i])
            self.assertAlmostEqual(avg_result, i, places=2)


    def test_random_traversal(self):

        oridata = torch.arange(self.max_index)
        dataset = TensorDataset(oridata)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        avg_list = [[] for _ in self.tensor_shapes]

        for i, data in enumerate(dataloader):
            idx = data[0]
            tensor_list = self.cache.retrieve(idx)
            for j in range(len(self.tensor_shapes)):
                avg = torch.mean(tensor_list[j])
                avg_list[j].append(avg.item())

        for i in range(len(self.tensor_shapes)):
            avg_result = np.mean(avg_list[i])
            self.assertAlmostEqual(avg_result, i, places=2)
