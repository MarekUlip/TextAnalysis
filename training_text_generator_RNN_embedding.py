from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from itertools import islice
from helper_functions import preprocess_sentence

class Training_Text_Generator_RNN_Embedding(Sequence):
    def __init__(self, filename, batch_size, num_of_texts,num_of_words, tokenizer: Tokenizer, delimeter, num_of_classes,max_len,start_point=0):
        self.filename = filename
        self.batch_size = batch_size
        self.num_of_texts = num_of_texts
        self.tokenizer = tokenizer
        self.delimeter = delimeter
        self.num_of_words = num_of_words
        self.num_of_classes = num_of_classes
        self.start_point = start_point
        self.max_len = max_len

    def __len__(self):
        """if self.start_point == 0:
            return np.ceil(self.num_of_texts/self.batch_size).astype(np.int)
        else:
            return np.ceil((self.num_of_texts) / self.batch_size).astype(np.int)"""
        return np.ceil(self.num_of_texts / self.batch_size).astype(np.int)

    def __getitem__(self, item):
        #print("returning batch for {} item".format(item))
        articles = []
        with open(self.filename, encoding='utf-8', errors='ignore') as csvfile:
            for row in islice(csv.reader(csvfile, delimiter=self.delimeter), self.start_point+item*self.batch_size,None):
                #print("getting item based on {}".format(item))
                articles.append([int(row[0]),row[1]])
                if len(articles) >= self.batch_size:
                    break

        articles = np.array(articles)
        if len(articles.shape) < 2:
            print("Working around...")
            articles = np.array([[0,"fgdssdgdsfgdsfgdsfg"],[1,"fgdssdgdsfgdsfgdsfg"]])
        labels = to_categorical(articles[:,0], num_classes=self.num_of_classes, dtype=np.uint8)
        features = self.tokenizer.texts_to_matrix(articles[:,1],mode="binary") #vectorize_sequences(self.tokenizer.texts_to_sequences(articles[:,1]),self.num_of_words).astype(np.uint8)
        articles = None
        return pad_sequences(features,maxlen=self.max_len),labels#np.reshape(features,(features.shape[0], 1, features.shape[1])), labels