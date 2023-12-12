import tensorflow as tf
from multi_head_attention import MultiHeadAttention

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.multi_head_self_attention = MultiHeadAttention()
        self.multi_head_encoder_attention = MultiHeadAttention()
        self.norm = tf.keras.layers.LayerNormalization()
        self.proj_output = tf.keras.layers.Dense(256)
        super().build(input_shape)
    
    def call(self, x):

        enc_output, output_embedding, mask = x

        self_attention = self.multi_head_self_attention((output_embedding, output_embedding, output_embedding), mask=mask)
        post_self_attention = self.norm(output_embedding + self_attention)
        enc_attention = self.multi_head_encoder_attention((post_self_attention, enc_output, enc_output))
        post_enc_attention = self.norm(enc_attention + post_self_attention)
        proj_out = self.proj_output(post_enc_attention)
        dec_output = self.norm(proj_out + post_enc_attention)
        return dec_output
    