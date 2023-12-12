import tensorflow as tf
from input_embedding import EmbeddingLayer

class MultiHeadAttention(tf.keras.layers.Layer):
    """
        C'est le multi-head attention utilisé ici.
        nb_head = nombre de têtes d'attention, dim doit être divisible par nb_head
    """
    def __init__(self, dim=256, nb_head = 8, **kwargs):
        self.dim = dim
        self.nb_head = nb_head
        self.head_dim = dim // nb_head
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.query_layer = tf.keras.layers.Dense(256) # on pourrait faire 512 aussi
        self.value_layer = tf.keras.layers.Dense(256)
        self.key_layer = tf.keras.layers.Dense(256)
        self.out_proj = tf.keras.layers.Dense(256)
        super().build(input_shape)

    def mask_softmax(self,x, mask):
        x_expe = tf.math.exp(x)
        x_expe_masked = x_expe * mask
        x_expe_sum = tf.reduce_sum(x_expe_masked, axis=-1)    
        x_expe_sum = tf.expand_dims(x_expe_sum, axis=-1)
        softmax = x_expe_masked / x_expe_sum
        return softmax
    
    def call(self, x, mask = None):

        in_query, in_key, in_value = x

        Q = self.query_layer(in_query)#1*5*256
        K = self.key_layer(in_key)
        V = self.value_layer(in_value)

        #création des têtes d'attentions
        batch_size = tf.shape(Q)[0]
        Q_seq_len = tf.shape(Q)[1]
        K_seq_len = tf.shape(K)[1]
        V_seq_len = tf.shape(V)[1]
        Q = tf.reshape(Q,[batch_size, Q_seq_len, self.nb_head, self.head_dim])#1*5*8*32
        K = tf.reshape(K,[batch_size, K_seq_len, self.nb_head, self.head_dim])
        V = tf.reshape(V,[batch_size, V_seq_len, self.nb_head, self.head_dim])

        Q = tf.transpose(Q,[0,2,1,3]) #on transpose les dimensions 2  et 1 (seq_len et nb_head) pour séparer les têtes
        K = tf.transpose(K,[0,2,1,3]) #1*8*5*32
        V = tf.transpose(V,[0,2,1,3])

        Q = tf.reshape(Q,[batch_size * self.nb_head, Q_seq_len, self.head_dim]) #8*5*32
        K = tf.reshape(K,[batch_size * self.nb_head, K_seq_len, self.head_dim])
        V = tf.reshape(V,[batch_size * self.nb_head, V_seq_len, self.head_dim])

        #Scaled-Dot Product Attention : Attention(Q, K, V ) = softmax(Qtranspose(K)/√dk)V 
        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK/tf.sqrt(256.)

        #Mask
        if mask is not None:
            QK = QK * mask
            softmax_QK = self.mask_softmax(QK, mask)
        else:
            softmax_QK = tf.nn.softmax(QK, axis =-1)
            
        attention = tf.matmul(softmax_QK, V)

        #concatenation
        attention = tf.reshape(attention,[batch_size, self.nb_head, Q_seq_len, self.head_dim])#1*8*5*32
        attention = tf.transpose(attention,[0,2,1,3])#1*5*8*32
        attention = tf.reshape(attention,[batch_size, Q_seq_len, self.nb_head * self.head_dim])#1*5*256

        #linear
        out_attention = self.out_proj(attention)

        return out_attention
    