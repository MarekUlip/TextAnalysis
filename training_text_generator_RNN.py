from aliaser import Sequence, to_categorical, Tokenizer
import csv
import numpy as np
from itertools import islice
from helper_functions import preprocess_sentence

"""def vectorize_sequences(sequences, dimension=20000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results"""

class Training_Text_Generator_RNN(Sequence):
    def __init__(self, filename, batch_size, num_of_texts,num_of_words, tokenizer: Tokenizer, delimeter, num_of_classes,start_point=0,is_predicting=False,preload_dataset=False,preprocess=False):
        self.filename = filename
        self.batch_size = batch_size
        self.num_of_texts = num_of_texts
        self.tokenizer = tokenizer
        self.delimeter = delimeter
        self.num_of_words = num_of_words
        self.num_of_classes = num_of_classes
        self.start_point = start_point
        self.is_predicting = is_predicting
        self.labels = []
        self.articles = []
        self.preload_dataset = preload_dataset
        self.preprocess = preprocess
        if preload_dataset:
            self.load_dataset()


    def __len__(self):
        """if self.start_point == 0:
            return np.ceil(self.num_of_texts/self.batch_size).astype(np.int)
        else:
            return np.ceil((self.num_of_texts) / self.batch_size).astype(np.int)"""
        return np.ceil(self.num_of_texts / self.batch_size).astype(np.int)
    def load_dataset(self):
        with open(self.filename, encoding='utf-8', errors='ignore') as csvfile:
            for row in csv.reader(csvfile, delimiter=self.delimeter):
                self.articles.append([int(row[0]),preprocess_sentence(row[1]) if self.preprocess else row[1]])
    def __getitem__(self, item):
        #print("returning batch for {} item".format(item))
        articles = []
        if self.preload_dataset:
            articles = self.articles[self.start_point+item*self.batch_size:self.start_point+(item+1)*self.batch_size]
        else:
            with open(self.filename, encoding='utf-8', errors='ignore') as csvfile:
                for row in islice(csv.reader(csvfile, delimiter=self.delimeter), self.start_point+item*self.batch_size,None):
                    #print("getting item based on {}".format(item))
                    articles.append([int(row[0]),preprocess_sentence(row[1]) if self.preprocess else row[1]])#preprocess_sentence(row[1])])
                    if len(articles) >= self.batch_size:
                        break

        articles = np.array(articles)
        if len(articles.shape) < 2:
            print("Working around...")
            articles = np.array([[0,"fgdssdgdsfgdsfgdsfg"]])
        if self.is_predicting:
            self.labels.extend(list(map(int,articles[:,0])))
        labels = to_categorical(articles[:,0], num_classes=self.num_of_classes, dtype=np.uint8)
        features = self.tokenizer.texts_to_matrix(articles[:,1],mode="binary") #vectorize_sequences(self.tokenizer.texts_to_sequences(articles[:,1]),self.num_of_words).astype(np.uint8)
        articles = None
        if self.is_predicting:
            return np.reshape(features,(features.shape[0], 1, features.shape[1]))
        else:
            return np.reshape(features,(features.shape[0], 1, features.shape[1])), labels#(1,features.shape[1],1)),labels