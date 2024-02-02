import pandas as pd
import pretty_midi
import tensorflow as tf
from symusic import Score

# from utils import midi_to_wav
from tokenizer import REMI, SEQ_LENGTH
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

    def __call__(self, score: Score(), max_length=SEQ_LENGTH):

        melody = tf.convert_to_tensor(self.tokenizer(score).ids[:SEQ_LENGTH])
        melody = tf.reshape(melody, (1, melody.shape[0]))
        print(melody)
        # melody = self.tokenizers.pt.tokenize(melody).to_tensor()

        encoder_input = melody

        # As the output language is English, initialize the output with the
        # English `[START]` token.
        # start_end = self.tokenizers.en.tokenize([""])[0]
        # start = start_end[0][tf.newaxis]
        # end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        # output_array = tf.TensorArray(
        #     dtype=tf.int64, size=0, dynamic_size=True
        # )
        output_array = tf.zeros((1, 63))

        for i in tf.range(max_length):
            # output = tf.transpose(output_array.stack())
            output = output_array
            print(encoder_input.shape, output.shape)
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

            # if predicted_id == end:
            #     break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        output_melody = self.tokenizer.tokens_to_midi(output.to_numpy())[
            0
        ]  # Shape: `()`

        # tokens = self.tokenizer

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return output_melody, output, attention_weights
