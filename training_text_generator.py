from neural_networks.aliaser import Sequence,to_categorical,Tokenizer
import csv
import numpy as np
from itertools import islice
from dataset_loader.dataset_helper import preprocess_sentence

"""def vectorize_sequences(sequences, dimension=20000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results"""

class Training_Text_Generator(Sequence):
    def __init__(self, filename, batch_size, num_of_texts,num_of_words, tokenizer: Tokenizer, delimeter, num_of_classes,start_point=0):
        self.filename = filename
        self.batch_size = batch_size
        self.num_of_texts = num_of_texts
        self.tokenizer = tokenizer
        self.delimeter = delimeter
        self.num_of_words = num_of_words
        self.num_of_classes = num_of_classes
        self.start_point = start_point

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
                articles.append([int(row[0]),preprocess_sentence(row[1])])
                if len(articles) >= self.batch_size:
                    break

        articles = np.array(articles)
        if len(articles.shape) < 2:
            print("Working around...")
            articles = np.array([[0,""]])
        labels = to_categorical(articles[:,0], num_classes=self.num_of_classes, dtype=np.uint8)
        features = self.tokenizer.texts_to_matrix(articles[:,1],mode="tfidf") #vectorize_sequences(self.tokenizer.texts_to_sequences(articles[:,1]),self.num_of_words).astype(np.uint8)
        articles = None
        return features, labels