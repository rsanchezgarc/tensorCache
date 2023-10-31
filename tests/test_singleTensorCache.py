import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from tensorCache.tensorCache import TensorCache

class TestTensorCache(TensorCache):
    def compute_idxs(self, idxs):
        values = torch.ones(len(idxs)) * torch.as_tensor(idxs)
        return torch.ones((len(idxs),) + self.tensor_shape) * values.unsqueeze(-1).unsqueeze(-1)

class TestSingleTensorPyTorchCache1(unittest.TestCase):
    def setUp(self):
        self.cache = TestTensorCache(3, (2, 2), 10, dtype=torch.float32, device="cpu")

    def test_insert_and_retrieve(self):
        expected_data = self.cache.compute_idxs(torch.tensor([1, 2]))

        self.cache.insert(torch.tensor([1, 2]), expected_data)

        retrieved_data = self.cache.retrieve(torch.tensor([1, 2]))
        self.assertTrue(torch.equal(retrieved_data, expected_data))

    def test_lru_replacement(self):

        ori_data = self.cache.compute_idxs(torch.tensor([1, 2, 3]))
        self.cache.insert(torch.tensor([1, 2, 3]), ori_data)

        self.cache.retrieve(torch.tensor([4, 5, 6]))

        retrieved_data = self.cache.retrieve(torch.tensor([1, 2, 3, 4, 5, 6]))
        expected_data = self.cache.compute_idxs(torch.tensor([1, 2, 3, 4, 5, 6]))
        self.assertTrue(torch.equal(retrieved_data, expected_data))

    def test_compute_idx(self):
        retrieved_data = self.cache.retrieve(torch.tensor([7]))
        expected_data = self.cache.compute_idxs(torch.tensor([7]))
        self.assertTrue(torch.equal(retrieved_data, expected_data))

    def test_mixed_retrieve(self):
        self.cache.insert(torch.tensor([1, 2]), self.cache.compute_idxs(torch.tensor([1, 2])))

        retrieved_data = self.cache.retrieve(torch.tensor([1, 8, 2, 9]))
        expected_data = self.cache.compute_idxs(torch.tensor([1, 8, 2, 9]))
        self.assertTrue(torch.equal(retrieved_data, expected_data))

    def test_empty_retrieve(self):
        retrieved_data = self.cache.retrieve(torch.tensor([1, 2]))
        expected_data = self.cache.compute_idxs(torch.tensor([1, 2]))
        self.assertTrue(torch.equal(retrieved_data, expected_data))

    def test_duplicate_indices(self):
        self.cache.insert(torch.tensor([1, 1, 2]), self.cache.compute_idxs(torch.tensor([1, 1, 2])))

        retrieved_data = self.cache.retrieve(torch.tensor([1, 2]))
        expected_data =  self.cache.compute_idxs(torch.tensor([1, 2]))
        self.assertTrue(torch.equal(retrieved_data, expected_data))
    def test_sequential_traversal(self):

        # Create a simple dataset

        oridata = torch.arange(10)
        dataset = TensorDataset(oridata)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Computing averages using the cache

        avg_list = []
        for i, data in enumerate(dataloader):
            idx = data[0]
            tensor = self.cache.retrieve(idx)
            avg = torch.mean(tensor)
            avg_list.append(avg.item())

        avg_result = np.mean(avg_list)
        self.assertAlmostEqual(avg_result, oridata.float().mean().item(), places=2)  # Updated expected value



    def test_random_traversal(self):

        # Create a simple dataset
        oridata = torch.arange(10)
        dataset = TensorDataset(oridata)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Computing averages using the cache
        avg_list = []
        for i, data in enumerate(dataloader):
            idx = data[0]
            tensor = self.cache.retrieve(idx)
            avg = torch.mean(tensor)
            avg_list.append(avg.item())

        avg_result = np.mean(avg_list)
        self.assertAlmostEqual(avg_result, oridata.float().mean().item(), places=2)


class TestSingleTensorPyTorchCache2(unittest.TestCase):

    def setUp(self):
        self.cache_size = 5
        self.max_index = 20
        self.cache = TestTensorCache(self.cache_size, (2, 2),
                                     self.max_index, dtype=torch.float32, device="cpu")


    def test_insert_and_retrieve_large(self):
        expected_data = self.cache.compute_idxs(torch.tensor([1, 2, 3]))
        self.cache.insert(torch.tensor([1, 2, 3]), expected_data)
        retrieved_data = self.cache.retrieve(torch.tensor([1, 2, 3]))
        self.assertTrue(torch.equal(retrieved_data, expected_data))



    def test_lru_replacement_large(self):

        ori_data = self.cache.compute_idxs(torch.tensor([1, 2, 3, 4, 5]))
        self.cache.insert(torch.tensor([1, 2, 3, 4, 5]), ori_data)
        self.cache.retrieve(torch.tensor([6, 7, 8]))

        retrieved_data = self.cache.retrieve(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
        expected_data = self.cache.compute_idxs(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertTrue(torch.equal(retrieved_data, expected_data))


    def test_random_traversal_large(self):

        # Create a simple dataset
        oridata = torch.arange(20)  # Dataset size increased
        dataset = TensorDataset(oridata)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Computing averages using the cache

        avg_list = []
        for i, data in enumerate(dataloader):
            idx = data[0]
            tensor = self.cache.retrieve(idx)
            avg = torch.mean(tensor)
            avg_list.append(avg.item())

        avg_result = np.mean(avg_list)
        self.assertAlmostEqual(avg_result, oridata.float().mean().item(), places=2)

    def test_arbitrary_indexing(self):

        num_iterations = 200
        cum_cache = 0
        cum_direct = 0
        for i in range(num_iterations):
            idx_subset =  torch.randint(0, self.max_index, size=(np.random.randint(0, self.cache_size),))
            cum_direct += self.cache.compute_idxs(idx_subset).sum()
            cum_cache += self.cache.retrieve(idx_subset).sum()

        self.assertAlmostEqual(cum_direct, cum_cache, places=2)


