import unittest

import numpy as np
import torch

from tensorCache.lookAheadCache import LookAheadTensorCache

logic_device="cpu"

class TestSingleTensorPyTorchCache2(unittest.TestCase):

    def setUp(self):
        class Lo(LookAheadTensorCache):
            def compute_idxs(self, idxs):
                # Mock computation: returns tensors filled with the index value
                return torch.stack(
                    [torch.full(self.tensor_shape, idx.item(), dtype=self.dtype, device=self.data_device) for idx in
                     idxs])

        cache = Lo(cache_size=4, tensor_shape=(1,), max_index=32, dtype=torch.float32, data_device="cpu",
                   logic_device=logic_device)
        return cache

    def test1(self):

        cache = self.setUp()
        out = cache[torch.tensor([5, 5, 6, 6, 7], device=logic_device)]
        try:
            out = cache[torch.tensor([5, 3, 6, 6, 7], device=logic_device)]
            self.fail("Error, it did not detect non monotonic sequence")
        except AssertionError:
            pass

    def __generic_test(self, tensor_list):
        cache = self.setUp()

        for indices in tensor_list:
            output = cache[indices]
            # print(indices.flatten())
            # print(output.flatten())
            self.assertTrue(torch.eq(indices.flatten(), output.cpu().flatten()).all())
            if len(indices) == cache.cache_size:
                self.assertTrue(cache.reverse_index_tracker[cache.reverse_index_tracker>=0].min() == indices.min(),
                                f"Error{cache.reverse_index_tracker.min(), indices.min()}")
            print(f"Retrieving indices: {indices}")
            print(f"Output: {output}")
            print(f"Cache state: {cache.cache}")
            print(f"Index tracker: {cache.index_tracker}\n")
            print(f"Rev Index tracker: {cache.reverse_index_tracker}\n")

    def test2(self):

        test_indices = [
            torch.tensor([5, 5, 6, 6, 7], device=logic_device),
            torch.tensor([10, 10, 11], device=logic_device),
            torch.tensor([15, 14, 15], device=logic_device),
            torch.tensor([20, 20, 20], device=logic_device),
            torch.tensor([20], device=logic_device),
            torch.tensor([20], device=logic_device),
        ]
        self.__generic_test(test_indices)

    def test3(self):

        test_indices = [
            torch.tensor([5, 5, 6, 6, 7], device=logic_device),
            torch.tensor([10, 10, 11], device=logic_device),
            torch.tensor([15, 14, 15], device=logic_device),
            torch.tensor([20, 20, 20], device=logic_device),
            torch.tensor([20, 23, 20, 23], device=logic_device),
            torch.tensor([28], device=logic_device),
            torch.tensor([30,31], device=logic_device),

        ]
        self.__generic_test(test_indices)

    def test4(self):
        test_indices = [
            torch.tensor([5, 5, 6, 6, 7], device=logic_device),
            torch.tensor([28], device=logic_device),
            torch.tensor([30, 31], device=logic_device),
            torch.tensor([32, 32], device=logic_device),
            torch.tensor([32], device=logic_device),

        ]
        self.__generic_test(test_indices)

    def test5(self):
        test_indices = [
            torch.tensor([30, 31,31,31], device=logic_device),
            torch.tensor([32, 32], device=logic_device),
            torch.tensor([32], device=logic_device),

        ]
        self.__generic_test(test_indices)

    def test6(self):
        test_indices = [
            torch.tensor([30, 31,31,32], device=logic_device),
            torch.tensor([32, 32], device=logic_device),
            torch.tensor([32], device=logic_device),

        ]
        self.__generic_test(test_indices)

    def test7(self):
        test_indices = [
            torch.tensor(range(4), device=logic_device),
            torch.tensor(range(4, 8), device=logic_device),
            torch.tensor(range(8, 12), device=logic_device),
            torch.tensor(range(29,33), device=logic_device),
            torch.tensor([32], device=logic_device),

        ]
        self.__generic_test(test_indices)

    def test8(self):
        test_indices = [
            torch.tensor(range(2), device=logic_device)
        ]
        self.__generic_test(test_indices)