import tensorflow as tf
from encoder_layer import EncoderLayer

class Encoder(tf.keras.layers.Layer):
    '''
    Appelle N fois l'EncoderLayer
    '''
    def __init__(self, nb_encoder, **kwargs):
        self.nb_encoder = nb_encoder
        super().__init__(**kwargs)
    
    def build(self, input_shape):

        self.encoder_layers = []
        for nb in range(self.nb_encoder):
            self.encoder_layers.append(
                EncoderLayer()
            )
        super().build(input_shape)
    
    def call(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
    
        return x