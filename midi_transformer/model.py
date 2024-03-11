import keras
import tensorflow as tf
from loss import masked_loss
from tokenizer import VOCAB_SIZE
from tranformer import (
    D_MODEL,
    DFF,
    DROPOUT_RATE,
    NUM_HEADS,
    NUM_LAYERS,
    Transformer,
)


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        config = {"warmup_steps": self.warmup_steps, "d_model": self.d_model}
        return config

    @classmethod
    def from_config(cls, config):
        d_model = config.pop("d_model")
        warmup_steps = config.pop("warmup_steps")
        return cls(d_model, warmup_steps, **config)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def create_model():
    learning_rate = CustomSchedule(D_MODEL)

    optimizer = keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    model = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=VOCAB_SIZE,
        target_vocab_size=VOCAB_SIZE,
        dropout_rate=DROPOUT_RATE,
    )

    model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy],
    )
    return model
