import keras
import tensorflow as tf
from keras import backend
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
    cross_entropy = backend.sparse_categorical_crossentropy(label, pred)
    return backend.mean(backend.exp(backend.mean(cross_entropy, axis=-1)))

