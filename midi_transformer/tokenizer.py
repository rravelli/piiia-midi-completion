import tensorflow as tf
import numpy as np
from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score
from tqdm import tqdm

MAX_TOKENS = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 64
SEQ_LENGTH = 64
TOKENIZER_CONFIG = TokenizerConfig(
    num_velocities=16, use_chords=True, use_programs=True
)
TOKENIZER = REMI(TOKENIZER_CONFIG)
VOCAB_SIZE = len(TOKENIZER.vocab)
MAX_NOTE = 100  # max number of notes taken from MIDI file


def split_labels(sequences):
    context = sequences[:SEQ_LENGTH]
    inputs = sequences[SEQ_LENGTH : 2 * SEQ_LENGTH - 1]
    labels = sequences[SEQ_LENGTH + 1 : 2 * SEQ_LENGTH]
    return (context, inputs), labels


def make_midi_batchesv2(
    file_paths: list[Path], num_files: int = 300, tokenizer=TOKENIZER
) -> tf.data.Dataset:
    """Generate a dataset from midi files

    Args:
        file_paths (list[Path]): List of paths of files to convert
        num_files (int, optional): Number of files to include in the dataset. Defaults to 300.

    Returns:
        DatasetV2: the generated dataset containing all midi files
    """
    all_notes = []
    for file in tqdm(file_paths[:num_files]):
        midi = Score(file)
        tokens = tokenizer(midi).ids
        for i in range(0, len(tokens) - 2 * SEQ_LENGTH, 2 * SEQ_LENGTH):
            sample = np.array(tokens[i : i + 2 * SEQ_LENGTH])
            all_notes.append(sample)

    train_notes = np.array(all_notes)

    sequences = tf.data.Dataset.from_tensor_slices(train_notes)
    seq_ds = sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    buffer_size = BUFFER_SIZE  # the number of items in the dataset
    train_ds = (
        seq_ds.shuffle(buffer_size)
        .batch(BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return train_ds
