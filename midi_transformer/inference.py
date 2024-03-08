import datetime
from os import environ, makedirs
from random import randint

import numpy as np
import pandas as pd
import pretty_midi
import tensorflow as tf
from dataset import download_maestro_dataset
from midi2audio import FluidSynth
from model import create_model
from symusic.factory import Score, ScoreFactory
from tokenizer import REMI, SEQ_LENGTH, TOKENIZER
from tqdm import tqdm
from tranformer import Transformer


def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str,
    instrument_name: str,
    velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:
    # Vérification pour un tableau pandas vide
    if notes.empty:
        raise ValueError(
            "Empty notes DataFrame. Cannot generate MIDI with no notes."
        )
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


class MIDIGenerator(tf.Module):
    def __init__(self, transformer: Transformer, tokenizer: REMI):
        self.transformer = transformer
        self.tokenizer = tokenizer

    def __call__(
        self,
        score: ScoreFactory,
        max_length=SEQ_LENGTH,
        recursions: int = 1,
        slice_start=True,
        offset: int = 0,
    ):
        if slice_start:
            melody = tf.convert_to_tensor(
                self.tokenizer(score).ids[SEQ_LENGTH + offset : 2 * SEQ_LENGTH]
            )
        else:
            melody = tf.convert_to_tensor(
                self.tokenizer(score).ids[-2 * SEQ_LENGTH :]
            )
        melody = tf.reshape(melody, (1, melody.shape[0]))

        encoder_input = melody
        final_output = melody.numpy()

        for _ in tqdm(range(recursions)):
            output_array = tf.TensorArray(
                dtype=tf.int64, size=0, dynamic_size=True
            )
            output_array = output_array.write(0, [0])
            encoder_input = tf.convert_to_tensor(final_output[-SEQ_LENGTH:])
            for i in tqdm(tf.range(max_length)):
                output = tf.transpose(output_array.stack())
                predictions = self.transformer(
                    (encoder_input, output), training=False
                )

                # Select the last token from the `seq_len` dimension.
                predictions = predictions[
                    :, -1:, :
                ]  # Shape `(batch_size, 1, vocab_size)`.

                predicted_id = tf.argmax(predictions, axis=-1)

                # Concatenate the `predicted_id` to the output which is given to the
                # decoder as its input.
                output_array = output_array.write(i + 1, predicted_id[0])

            output = tf.transpose(output_array.stack())
            final_output = np.concatenate(
                (final_output, output.numpy()), axis=1
            )

        input_ = self.tokenizer(score).ids
        if slice_start:
            input_ = input_[: 2 * SEQ_LENGTH]

        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return input_, final_output[0], attention_weights


class MIDIGenerator2(tf.Module):
    def __init__(self, transformer: Transformer, tokenizer: REMI):
        self.transformer = transformer
        self.tokenizer = tokenizer

    def __call__(
        self,
        score: ScoreFactory,
        recursions: int = 1,
        offset: int = 0,
    ):

        melody = self.tokenizer(score).ids[offset : 4 * SEQ_LENGTH + offset]

        data_list = melody
        output = []
        for _ in tqdm(range(recursions)):
            data = tf.constant([data_list])

            context_array = data[:, -2 * SEQ_LENGTH : SEQ_LENGTH]
            input_array = data[:, -SEQ_LENGTH:]
            predictions = self.transformer(
                (
                    context_array,
                    input_array,
                ),
                training=False,
            )

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[
                :, -1:, :
            ]  # Shape `(batch_size, 1, vocab_size)`.
            predicted_id = tf.argmax(predictions, axis=-1)
            token = int(predicted_id[0, 0])
            data_list.append(token)
            output.append(token)

        return melody, output


def generate_sample(
    input_file: str,
    recursions=1,
    output_dir="generated_samples",
    weights_path: str = "training_data/training_2/cp.ckpt",
):
    score = Score(input_file)
    # init model
    model = create_model()
    model.load_weights(weights_path)
    # offset = randint(1, 1000)
    # generate midi
    input, output = MIDIGenerator2(model, TOKENIZER)(
        score, recursions=recursions
    )

    date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    sample_dir = f"{output_dir}/{date}"
    makedirs(sample_dir, exist_ok=True)
    # convert to midi
    TOKENIZER(input).dump_midi(f"{sample_dir}/input.mid")
    TOKENIZER(output).dump_midi(f"{sample_dir}/output.mid")
    TOKENIZER(list(input) + list(output)).dump_midi(f"{sample_dir}/final.mid")
    with open(f"{sample_dir}/.gitignore", "w") as f:
        f.write("*")
    # convert to audio
    FluidSynth().midi_to_audio(
        f"{sample_dir}/final.mid", f"{sample_dir}/final.wav"
    )
    FluidSynth().midi_to_audio(
        f"{sample_dir}/output.mid", f"{sample_dir}/output.wav"
    )


if __name__ == "__main__":
    # set gpu to false
    environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # get midis
    _, _, examples = download_maestro_dataset()
    file = examples[randint(0, len(examples) - 1)]
    # file = "./Pop 1 - C major.mid"
    # generate
    generate_sample(
        file,
        recursions=64 * 8,
        weights_path="training_data/training_2/cp.ckpt",
    )
