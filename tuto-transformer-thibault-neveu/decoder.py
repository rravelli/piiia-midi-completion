import tensorflow as tf
from decoder_layer import DecoderLayer

class Decoder(tf.keras.layers.Layer):
    '''
    Appelle N fois le DecoderLayer
    '''
    def __init__(self, nb_decoder, **kwargs):
        self.nb_decoder = nb_decoder
        super().__init__(**kwargs)
    
    def build(self, input_shape):

        self.decoder_layers = []
        for nb in range(self.nb_decoder):
            self.decoder_layers.append(
                DecoderLayer()
            )
        super().build(input_shape)
    
    def call(self, x):

        enc_output, output_embedding, mask = x

        dec_output = output_embedding
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer((enc_output, dec_output, mask))
    
        return dec_output