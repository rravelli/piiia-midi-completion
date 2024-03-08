import keras
import tensorflow as tf
import keras.backend as K
from tokenizer import VOCAB_SIZE


def masked_loss(label, pred):
    mask = label != 0
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def perplexity(label, pred):
    mask = label != 0
    cross_entropy = K.sparse_categorical_crossentropy(label, pred)
    perplexity = K.exp(cross_entropy)

    mask = tf.cast(mask, dtype=perplexity.dtype)
    perplexity *= mask

    perplexity = tf.reduce_sum(perplexity) / tf.reduce_sum(mask)

    return perplexity
