import pathlib
import glob
from src.dataset import download_maestro_dataset
import pandas as pd


def test_download_maestro_dataset():
    download_maestro_dataset()
    data_dir = pathlib.Path("data/maestro-v3.0.0")
    assert data_dir.exists()
    filenames = glob.glob(str(data_dir / "**/*.mid*"))
    print("Number of files:", len(filenames))


test_download_maestro_dataset()
# TODO : fix
