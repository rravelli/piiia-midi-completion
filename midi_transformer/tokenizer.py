import tensorflow as tf
import keras

MAX_TOKENS = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 64


# Download tokenizer model
def download_tokenizer():
    model_name = "ted_hrlr_translate_pt_en_converter"
    keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",  # noqa
        cache_dir=".",
        cache_subdir="",
        extract=True,
    )
    return tf.saved_model.load(model_name)


def prepare_batch(pt, en, tokenizer):
    """The following function takes batches of text as input, and converts
    them to a format suitable for training.

    1. It tokenizes them into ragged batches.
    2. It trims each to be no longer than `MAX_TOKENS`.
    3. It splits the target (English) tokens into inputs and labels. These are
    shifted by one step so that at each input location the `label` is the id
    of the next token.
    4. It converts the `RaggedTensor`s to padded dense `Tensor`s.
    5. It returns an `(inputs, labels)` pair.
    """
    pt = tokenizer.pt.tokenize(pt)  # Output is ragged.
    pt = pt[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizer.en.tokenize(en)
    en = en[:, : (MAX_TOKENS + 1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

    return (pt, en_inputs), en_labels


def make_batches(ds: tf.data.Dataset, tokenizer):
    """The function below converts a dataset of text examples into data of
    batches for training.

    1. It tokenizes the text, and filters out the sequences that are too long.
       (The `batch`/`unbatch` is included because the tokenizer is much more
       efficient on large batches).
    2. The `cache` method ensures that that work is only executed once.
    3. Then `shuffle` and, `dense_to_ragged_batch` randomize the order and
    assemble batches of examples.
    4. Finally `prefetch` runs the dataset in parallel with the model to
    ensure that data is available when needed. See [Better performance with
    the [tf.data](https://www.tensorflow.org/guide/data_performance.ipynb)
    for details.
    """
    return (
        ds.shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(lambda pt, en: prepare_batch(pt, en, tokenizer), tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
