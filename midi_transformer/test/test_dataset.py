import pathlib
import os
from dataset import download_maestro_dataset
from unittest import TestCase


class TestDataset(TestCase):
    def test_download_maestro_dataset(self):
        train_path, test_path, validation_path = download_maestro_dataset()
        data_dir = pathlib.Path("data/maestro-v3.0.0")
        self.assertTrue(data_dir.exists())
        self.assertEqual(len(train_path) + len(test_path) + len(validation_path), 1276)
        abs_path = os.path.abspath(data_dir)
        self.assertTrue(str(train_path[0]).startswith(abs_path))
        self.assertTrue(str(train_path[0]).endswith(".midi"))
