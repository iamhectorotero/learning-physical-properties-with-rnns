import unittest
from isaac import utils
import torch
import shutil
import os
import warnings

class TestGetCudaDeviceIfAvailable(unittest.TestCase):
    def test_cuda_device_is_returned(self):
        device_returned = utils.get_cuda_device_if_available()
        if os.environ.get("CUDA_VISIBLE_DEVICES", default="") == "":
            warnings.warn("No Cuda device available to run this test.")
            self.assertEqual(device_returned, torch.device("cpu"))
        else:
            self.assertEqual(device_returned, torch.device("cuda:0"))

    def test_cpu_is_returned(self):
        _visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", default="")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        device_returned = utils.get_cuda_device_if_available()
        self.assertEqual(device_returned, torch.device("cpu"))

        os.environ["CUDA_VISIBLE_DEVICES"] = _visible_devices

class TestCreateDirectory(unittest.TestCase):
    dir_path = "temporary_directory"
    filename = "temporary_file"
    file_path = os.path.join(dir_path, filename)

    def test_without_previously_existing_directory(self):
        utils.create_directory(self.dir_path)
        self.assertTrue(os.path.exists(self.dir_path))

        os.rmdir(self.dir_path)

    def test_deleting_previously_existing_directory(self):
        os.makedirs(self.dir_path)
        f = open(self.file_path, "w+")
        f.close()

        utils.create_directory(self.dir_path, delete_if_exists=True)

        self.assertTrue(os.path.exists(self.dir_path))
        self.assertFalse(os.path.exists(self.file_path))

        os.rmdir(self.dir_path)

    def test_respecting_previously_existing_directory(self):
        os.makedirs(self.dir_path)
        f = open(self.file_path, "w+")
        f.close()

        utils.create_directory(self.dir_path, delete_if_exists=False)

        self.assertTrue(os.path.exists(self.dir_path))
        self.assertTrue(os.path.exists(self.file_path))

        shutil.rmtree(self.dir_path)

if __name__ == "__main__":
    unittest.main()

