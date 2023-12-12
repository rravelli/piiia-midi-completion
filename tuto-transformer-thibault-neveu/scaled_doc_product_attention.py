import tensorflow as tf

class ScaledDotProductAttention(tf.keras.layers.Layer):
    """
        C'est le scaled_dot product attention utilisé pour le multi-head attention ici. Façonné selon la formule : 
        Attention(Q, K, V ) = softmax(Qtranspose(K)/√dk)V 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.query_layer = tf.keras.layers.Dense(256) # on pourrait faire 512 aussi
        self.value_layer = tf.keras.layers.Dense(256)
        self.key_layer = tf.keras.layers.Dense(256)
        super().build(input_shape)
    
    def call(self, x):
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK/tf.sqrt(256.)

        softmax_QK = tf.nn.softmax(QK, axis =-1)
        #print(tf.reduce_sum(softmax_QK, axis=-1)) # check, on doit obtenir quasi 1 partout

        attention = tf.matmul(softmax_QK, V)

        return attention
    