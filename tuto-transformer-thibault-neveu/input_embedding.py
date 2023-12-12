import tensorflow as tf

class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, nb_token, **kwargs):
        self.nb_token = nb_token
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.word_embedding = tf.keras.layers.Embedding(self.nb_token, 256)
        super().build(input_shape)
    
    def call(self, x):
        embed = self.word_embedding(x)
        return embed