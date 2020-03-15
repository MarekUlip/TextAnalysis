import os
import numpy as np
from dataset_loader.dataset_helper import get_root_folder

embeddings_path = get_root_folder() + "\\embeddings\\"
def get_embedding_matrix(max_words, embedding_dim, word_index):
    embeddings_index = {}
    f = open(os.path.join(embeddings_path, 'glove.6B.{}d.txt'.format(embedding_dim)),encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix