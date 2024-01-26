import tensorflow as tf
import keras
import pretty_midi
import collections
import pandas as pd
import numpy as np
from pathlib import Path

MAX_TOKENS = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 64
VOCAB_SIZE = 128
SEQ_LENGTH = 25
KEY_ORDER = ["pitch", "step", "duration"]
MAX_NOTE = 100  # max number of notes taken from MIDI file


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """
        Converts a MIDI file to a Dataframe
    Args:
        midi_file (str): path to the MIDI file

    Returns:
        pd.DataFrame: Dataframe of the notes in the MIDI file, with 5 columns :
                        - pitch : the pitch of the note
                        - start : timestamp for when the note starts
                        - end : timestamp for when the note ends
                        - step : time elapsed from the previous note or start of the track
                        - duration : duration from start to end of the note in seconds
    """
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes[:MAX_NOTE], key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes["pitch"].append(note.pitch)
        notes["start"].append(start)
        notes["end"].append(end)
        notes["step"].append(start - prev_start)
        notes["duration"].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size=128,
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    # seq_length = seq_length + 1

    # Take 1 extra for the labels
    windows = dataset.window(2 * seq_length, shift=1, stride=1, drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    def flatten(ds: tf.data.Dataset):
        return ds.batch(2 * seq_length, drop_remainder=True)

    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        context = sequences[:seq_length]
        inputs = sequences[seq_length : 2 * seq_length - 1]
        labels = sequences[seq_length + 1 : 2 * seq_length]
        # labels = {key: labels_dense[i] for i, key in enumerate(KEY_ORDER)}
        return (context[:, 0], inputs[:, 0]), labels[:, 0]

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def make_midi_batches(
    file_paths: list[Path],
    num_files: int = 300,
) -> tf.data.Dataset:
    """Generate a dataset from midi files

    Args:
        file_paths (list[Path]): List of paths of files to convert
        num_files (int, optional): Number of files to include in the dataset. Defaults to 300.

    Returns:
        DatasetV2: the generated dataset containing all midi files
    """
    all_notes = []
    for f in file_paths[:num_files]:
        notes = midi_to_notes(str(f))
        all_notes.append(notes)

    all_notes = pd.concat(all_notes)

    train_notes = np.stack([all_notes[key] for key in KEY_ORDER], axis=1)

    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    notes_ds.element_spec

    seq_ds = create_sequences(notes_ds, SEQ_LENGTH, VOCAB_SIZE)

    buffer_size = len(all_notes) - SEQ_LENGTH  # the number of items in the dataset
    train_ds = (
        seq_ds.shuffle(buffer_size)
        .batch(BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return train_ds
