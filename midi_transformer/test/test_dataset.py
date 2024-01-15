import pathlib
import glob
from dataset import download_maestro_dataset
from unittest import TestCase


class TestDataset(TestCase):
    def test_download_maestro_dataset(self):
        download_maestro_dataset()
        data_dir = pathlib.Path("data/maestro-v3.0.0")
        assert data_dir.exists()
        filenames = glob.glob(str(data_dir / "**/*.mid*"))
        print("Number of files:", len(filenames))
