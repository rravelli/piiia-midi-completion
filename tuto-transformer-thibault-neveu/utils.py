import numpy as np

# définir un vocabulaire, le nombre de tokens avec lesquels on va travailler
# il faut minimiser le nb de token possible, ici, exemple de bourrin

def  get_vocabulary(sequences:[[]]):
    token_to_info = {}
    for sequence in sequences:
        for word in sequence :
            if word not in token_to_info:
                token_to_info[word] = len(token_to_info) # on donne au mot un id unique

    return token_to_info

def sequence_to_token(sequences, voc):
    for sequence in sequences:
        for i,word in enumerate(sequence) :
            sequence[i]=voc[word]
    return np.array(sequences)
