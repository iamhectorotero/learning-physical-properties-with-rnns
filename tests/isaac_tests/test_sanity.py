import unittest
from isaac import sanity
import torch
from torch.utils.data import TensorDataset, DataLoader

class TestClassProportions(unittest.TestCase):
    def test_standard_case(self):
        X = torch.arange(7 * 10, dtype=torch.double).view(7, 10)
        Y = torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.long)
        tensor_dataset = TensorDataset(X, Y)
        data_loader = DataLoader(tensor_dataset)

        counts, majority_class_proportion = sanity.class_proportions(data_loader)

        self.assertListEqual(list(counts), [3, 2, 2])
        self.assertAlmostEqual(majority_class_proportion, 3./7)

    def test_only_one_class(self):
        X = torch.arange(4 * 10, dtype=torch.double).view(4, 10)
        Y = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        tensor_dataset = TensorDataset(X, Y)
        data_loader = DataLoader(tensor_dataset)

        counts, majority_class_proportion = sanity.class_proportions(data_loader)

        self.assertListEqual(list(counts), [4])
        self.assertAlmostEqual(majority_class_proportion, 1.)

    def test_only_equiprobable_classes(self):
        X = torch.arange(6 * 10, dtype=torch.double).view(6, 10)
        Y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        tensor_dataset = TensorDataset(X, Y)
        data_loader = DataLoader(tensor_dataset)

        counts, majority_class_proportion = sanity.class_proportions(data_loader)

        self.assertListEqual(list(counts), [2, 2, 2])
        self.assertAlmostEqual(majority_class_proportion, 2./6)

if __name__ == "__main__":
    unittest.main()
