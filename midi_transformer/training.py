import tensorflow as tf
import keras
from dataset import download_dataset
from tokenizer import download_tokenizer, make_batches, VOCAB_SIZE, SEQ_LENGTH
import tensorflow_text as text  # noqa
from tranformer import (
    Transformer,
    D_MODEL,
    NUM_LAYERS,
    NUM_HEADS,
    DROPOUT_RATE,
    DFF,
)
import pickle
from loss import mse_with_positive_pressure

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_examples, val_examples = download_dataset()
tokenizers = download_tokenizer()
train_batches = make_batches(train_examples, tokenizers)
val_batches = make_batches(val_examples, tokenizers)


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

optimizer = keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


checkpoint_path = "training_1/cp.ckpt"

# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

history = keras.callbacks.History()

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=VOCAB_SIZE,
    target_vocab_size=VOCAB_SIZE,
    dropout_rate=DROPOUT_RATE,
)

inputs = keras.Input((SEQ_LENGTH, 3))
x = transformer(inputs)

outputs = {
    "pitch": keras.layers.Dense(128, name="pitch")(x),
    "step": keras.layers.Dense(1, name="step")(x),
    "duration": keras.layers.Dense(1, name="duration")(x),
}

model = keras.Model(inputs, outputs)

print(model.summary())

model.compile(
    loss={
        "pitch": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "step": mse_with_positive_pressure,
        "duration": mse_with_positive_pressure,
    },
    loss_weights={
        "pitch": 0.05,
        "step": 1.0,
        "duration": 1.0,
    },
    optimizer=optimizer,
    metrics=[masked_accuracy],
)

model.fit(
    train_batches,
    epochs=20,
    validation_data=val_batches,
    callbacks=[cp_callback, history],
)

# sauvegarde de history
with open("history.pkl", "wb") as file:
    pickle.dump(history.history, file)
