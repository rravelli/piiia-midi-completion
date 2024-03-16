import keras
import keras.backend as K
import tensorflow as tf


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
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    cross_entropy = loss_object(label, pred)

    mask = tf.cast(mask, dtype=cross_entropy.dtype)
    cross_entropy *= mask
    step1 = K.mean(cross_entropy, axis=-1)
    step2 = K.exp(step1)
    perplexity = K.mean(step2)

    return perplexity
