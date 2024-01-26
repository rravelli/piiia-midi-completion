import pandas as pd
import pretty_midi
import tensorflow as tf
import pickle
from transformers import TFAutoModel
from tokenizer import midi_to_notes
from utils import midi_to_wav
from tokenizer import SEQ_LENGTH, create_sequences, KEY_ORDER, VOCAB_SIZE
import numpy as np


def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str,
    instrument_name: str,
    velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:
    # Vérification pour un tableau pandas vide
    if notes.empty:
        raise ValueError("Empty notes DataFrame. Cannot generate MIDI with no notes.")
    # Vérification pour des durées de note négatives
    if (notes["duration"] < 0).any():
        raise ValueError("Note duration must be positive.")
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note["step"])
        end = float(start + note["duration"])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note["pitch"]),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


def generate_music(midi_file_path, transformer_path, name_output):
    transformer = tf.keras.models.load_model(transformer_path + "midi_transformer")

    notes_midi = midi_to_notes(midi_file_path)

    generator = MIDIGenerator(transformer)
    generated_midi, generated_tokens, attention_weights = generator(
        tf.constant(notes_midi)
    )

    midi_to_wav(generated_midi, name_output)


import glob
import pathlib


def lets_try_this_baby():
    data_dir = pathlib.Path("data/maestro-v2.0.0")
    filenames = glob.glob(str(data_dir / "**/*.mid*"))
    midi_sample = filenames[1000]
    output = "music_original_test"
    midi_to_wav(midi_sample, output)

    generate_music(midi_sample, "", "music_generated_test")


class MIDIGenerator(tf.Module):
    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, melody: pd.DataFrame, max_length=SEQ_LENGTH):
        notes = np.stack([melody[key] for key in KEY_ORDER], axis=1)
        notes_ds = tf.data.Dataset.from_tensor_slices(notes)

        melody = create_sequences(notes_ds, SEQ_LENGTH, VOCAB_SIZE)
        # melody = self.tokenizers.pt.tokenize(melody).to_tensor()

        encoder_input = melody

        # As the output language is English, initialize the output with the
        # English `[START]` token.
        start_end = self.tokenizers.en.tokenize([""])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

        tokens = tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights
