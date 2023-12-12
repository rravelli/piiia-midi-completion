import tensorflow as tf
from utils import get_vocabulary, sequence_to_token
from input_embedding import EmbeddingLayer
from scaled_doc_product_attention import ScaledDotProductAttention
from encoder import Encoder
from multi_head_attention import MultiHeadAttention
from decoder import Decoder


INPUT = [["Salut", "comment", "ça", "va", "?"]]
OUTPUT = [["<START>","Hi", "how","are", "you", "?"]]

input_voc = get_vocabulary(INPUT)
output_voc = get_vocabulary(OUTPUT)

# token de début et de fin
input_voc["<START>"] = len(input_voc)
input_voc["<END>"] = len(input_voc)
input_voc["<PAD>"] = len(input_voc) # ajouter des tokens factices pour pouvoir traiter plusieurs phrases de longueurs diffs en une fois
output_voc["<END>"] = len(output_voc)
output_voc["<PAD>"] = len(output_voc) # ajouter des tokens factices pour pouvoir traiter plusieurs phrases de longueurs diffs en une fois

input_seq = sequence_to_token(INPUT, input_voc)
output_seq = sequence_to_token(OUTPUT, output_voc)

def get_transformer_model(output_voc):
    
    input_tokens = tf.keras.Input(shape=(5))
    output_tokens = tf.keras.Input(shape=(6))

    #Positional Encoding
    input_pos_encoding = EmbeddingLayer(nb_token=5)(tf.range(5))
    output_pos_encoding = EmbeddingLayer(nb_token=6)(tf.range(6))

    #Retrieve embedding
    input_embedding = EmbeddingLayer(nb_token=5)(input_tokens)
    output_embedding = EmbeddingLayer(nb_token=6)(output_tokens)

    #Add positional encoding
    input_embedding = input_embedding + input_pos_encoding
    output_embedding = output_embedding + output_pos_encoding

    #Encoder
    enc_output = Encoder(nb_encoder=6)(input_embedding)

    #Mask
    mask = tf.sequence_mask(tf.range(6) + 1, 6)
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis = 0)

    #Decoder
    dec_output = Decoder(nb_decoder=6)((enc_output, output_embedding, mask))

    #Predictions
    out_pred = tf.keras.layers.Dense(len(output_voc))(dec_output)
    predictions = tf.nn.softmax(out_pred, axis = -1)



    model = tf.keras.Model([input_tokens, output_tokens], predictions)
    model.summary()
    return model


transformer = get_transformer_model(output_voc)
out = transformer((input_seq, output_seq))
print(out.shape)

#ajouter loss et tout pour entraîner + vérifier car plus de paramètres sur dec que enc alors que lui non