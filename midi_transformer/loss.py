import keras
import tensorflow as tf
from keras import backend as K
from tokenizer import VOCAB_SIZE


def masked_loss(label, pred):
    print(label, pred)
    print(VOCAB_SIZE)
    mask = label != 0
    print(mask)
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def perplexity(label, pred):
    cross_entropy = K.sparse_categorical_crossentropy(
        label, pred, from_logits=True
    )
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))


def perplexity1(label, pred):
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    loss = loss_object(label, pred)
    return tf.math.exp(loss)


def perplexity2(label, pred):
    cross_entropy = K.sparse_categorical_crossentropy(
        label, pred, from_logits=True
    )
    perplexity = K.exp(cross_entropy)
    return perplexity


def perplexity3(label, pred):
    return K.exp(
        K.mean(
            K.sparse_categorical_crossentropy(label, pred, from_logits=True)
        )
    )
