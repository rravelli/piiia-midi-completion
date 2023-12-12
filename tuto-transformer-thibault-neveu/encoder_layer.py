import tensorflow as tf
from multi_head_attention import MultiHeadAttention

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.multi_head_attention = MultiHeadAttention()
        self.norm = tf.keras.layers.LayerNormalization()
        self.dense_out = tf.keras.layers.Dense(256)
        super().build(input_shape)
    
    def call(self, x):
        attention = self.multi_head_attention((x, x, x))
        post_attention = self.norm(attention + x)
        x = self.dense_out(post_attention)
        enc_output = self.norm(x + post_attention)
        return enc_output