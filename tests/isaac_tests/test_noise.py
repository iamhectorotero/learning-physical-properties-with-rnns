import unittest
import torch
import torch.utils.data
import numpy as np

import isaac.noise

class TestAddNoiseToDataloader(unittest.TestCase):
    @staticmethod
    def create_loader(n_features, n_classes, n_examples=10, seq_length=100, multibranch=False):

        X = np.random.rand(n_examples, seq_length, n_features).astype('f')

        if multibranch:
            Y = np.random.randint(low=0, high=n_classes, size=(n_examples, 2))
        else:
            Y = np.random.randint(low=0, high=n_classes, size=n_examples)

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        tensor_dataset = torch.utils.data.TensorDataset(X, Y)
        val_loader = torch.utils.data.DataLoader(tensor_dataset, shuffle=False)

        return val_loader

    def test_default_stddev(self):
        dl = self.create_loader(10, 2, 25000)

        dl_with_noise = isaac.noise.add_noise_to_dataloader(dl)

        # check dimensions
        self.assertEqual(dl.dataset.tensors[0].shape, dl_with_noise.dataset.tensors[0].shape)
        self.assertEqual(dl.dataset.tensors[1].shape, dl_with_noise.dataset.tensors[1].shape)

        # check Ys are the same
        self.assertTrue(torch.all(torch.eq(dl.dataset.tensors[1], dl_with_noise.dataset.tensors[1])))
        self.assertAlmostEqual(float(dl.dataset.tensors[0].mean()), float(dl_with_noise.dataset.tensors[0].mean()), 3)
        self.assertFalse(torch.all(torch.eq(dl.dataset.tensors[0], dl_with_noise.dataset.tensors[0])))

    def test_high_stddev(self):
        # Small sample sizes or high standard deviations can lead to the means not being the same.

        dl = self.create_loader(10, 2, 25000)

        dl_with_noise = isaac.noise.add_noise_to_dataloader(dl, noise_deviation=10.)

        # check dimensions
        self.assertEqual(dl.dataset.tensors[0].shape, dl_with_noise.dataset.tensors[0].shape)
        self.assertEqual(dl.dataset.tensors[1].shape, dl_with_noise.dataset.tensors[1].shape)

        # check Ys are the same
        self.assertTrue(torch.all(torch.eq(dl.dataset.tensors[1], dl_with_noise.dataset.tensors[1])))
        self.assertNotAlmostEqual(float(dl.dataset.tensors[0].mean()), float(dl_with_noise.dataset.tensors[0].mean()))

    def test_reproducibility(self):

        dl = self.create_loader(10, 2, 25000)
        dl_with_noise = isaac.noise.add_noise_to_dataloader(dl, noise_deviation=10., seed=72)
        dl_with_noise2 = isaac.noise.add_noise_to_dataloader(dl, noise_deviation=10., seed=72)

        self.assertTrue(torch.allclose(dl_with_noise.dataset.tensors[0], dl_with_noise2.dataset.tensors[0]))


    def test_variability_by_setting_different_seeds(self):
        dl = self.create_loader(10, 2, 25000)
        dl_with_noise = isaac.noise.add_noise_to_dataloader(dl, noise_deviation=10., seed=5)
        dl_with_noise2 = isaac.noise.add_noise_to_dataloader(dl, noise_deviation=10., seed=10)

        self.assertFalse(torch.allclose(dl_with_noise.dataset.tensors[0], dl_with_noise2.dataset.tensors[0]))


if __name__ == "__main__":
    unittest.main()
