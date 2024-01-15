# import tensorflow_datasets as tfds
import pathlib
import tensorflow as tf
import pandas as pd
import os


"""def download_dataset():
    examples, metadata = tfds.load(
        "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
    )

    return examples["train"], examples["validation"]"""


def download_maestro_dataset():
    """
    Downloads MAESTRO MIDI dataset in a data/maestro-v.3.0.0-midi
    """
    data_dir = pathlib.Path("data/maestro-v3.0.0")
    if not data_dir.exists():
        tf.keras.utils.get_file(
            "maestro-v3.0.0-midi.zip",
            origin="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
            extract=True,
            cache_dir=".",
            cache_subdir="data",
        )
        os.remove("data/maestro-v3.0.0-midi.zip")

    csv_dir = pathlib.Path("data/maestro-v3.0.0/maestro-v3.0.0.csv")
    midi_df = pd.read_csv(csv_dir, sep=",")
    train_df = midi_df.loc[midi_df["split"] == "train"]
    test_df = midi_df.loc[midi_df["split"] == "test"]
    validation_df = midi_df.loc[midi_df["split"] == "validation"]

    # TODO : list de pathlib.path de chaque truc
