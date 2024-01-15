import tensorflow_datasets as tfds
import pathlib
import tensorflow as tf
import pandas as pd
import os


def download_dataset():
    examples, metadata = tfds.load(
        "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
    )

    return examples["train"], examples["validation"]


def download_maestro_dataset():
    """
    Downloads MAESTRO MIDI dataset in a data/maestro-v.3.0.0-midi.
    Returns a train/test/validation split of the MIDIs.

    Returns:
        tuple[list[pathlib.Path]]: tuple of three lists of paths to the MIDIs.
                                   First contains the paths of the MIDIs for the train dataset,
                                   second contains the paths of the MIDIs for the test dataset,
                                   third contains the paths of the MIDIs for the validation dataset,

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

    train_df = midi_df["midi_filename"].loc[midi_df["split"] == "train"]
    test_df = midi_df["midi_filename"].loc[midi_df["split"] == "test"]
    validation_df = midi_df["midi_filename"].loc[midi_df["split"] == "validation"]

    train_midi = extract_path(train_df, data_dir)
    test_midi = extract_path(test_df, data_dir)
    validation_midi = extract_path(validation_df, data_dir)

    return train_midi, test_midi, validation_midi


def extract_path(dataset: pd.Series, data_dir: pathlib.Path):
    """
    From a dataset of paths, returns a list the absolute paths corresponding.

    Args:
        dataset (pd.Series): series containing paths to MIDIs
        data_dir (pathlib.Path): path to the folder where all MIDIs are

    Returns:
        list[pathlib.Path] : list of all absolute paths to MIDIs
    """
    list_path = []

    for path in dataset:
        list_path.append(pathlib.Path(os.path.abspath(data_dir.joinpath(path))))

    return list_path
