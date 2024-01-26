import pandas as pd
import pretty_midi
import tensorflow as tf
import pickle
from transformers import TFAutoModel
from tokenizer import midi_to_notes
from utils import midi_to_wav

SEQ_LENGTH = 25


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
    transformer = TFAutoModel.from_pretrained("transformer")
    transformer.load_weights(transformer_path + "model_transformer.ckpt")

    notes_midi = midi_to_notes(midi_file_path)

    generator = Generator(transformer)
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
