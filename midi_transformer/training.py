import tensorflow as tf
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

import tensorflow_text
from dataset import download_dataset
from tokenizer import download_tokenizer, make_batches


train_examples, val_examples = download_dataset()
tokenizers = download_tokenizer()
train_batches = make_batches(train_examples, tokenizers)
val_batches = make_batches(val_examples, tokenizers)

# import tensorflow_text as text
from tranformer import (
    Transformer,
    D_MODEL,
    NUM_LAYERS,
    NUM_HEADS,
    DEOPOUT_RATE,
    DFF,
)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=DEOPOUT_RATE,
)

transformer.compile(
    loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
)

transformer.fit(train_batches, epochs=20, validation_data=val_batches)
