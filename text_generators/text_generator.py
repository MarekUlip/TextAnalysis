from keras.preprocessing.text import Tokenizer
from keras.utils import Sequence
import numpy as np
import csv
from itertools import islice
from helper_functions import preprocess_sentence

class TextGenerator(Sequence):
    def __init__(self, filename, batch_size, num_of_texts, num_of_words, tokenizer: Tokenizer, delimeter,
                 num_of_classes, max_len=None, start_point=0, preprocess=False):
        self.filename = filename
        self.batch_size = batch_size
        self.num_of_texts = num_of_texts
        self.tokenizer = tokenizer
        self.delimeter = delimeter
        self.num_of_words = num_of_words
        self.num_of_classes = num_of_classes
        self.start_point = start_point
        self.max_len = max_len
        self.preprocess = preprocess
        self.articles = None

    def __len__(self):
        return np.ceil(self.num_of_texts / self.batch_size).astype(np.int)

    def __getitem__(self, item):
        self.articles = []
        with open(self.filename, encoding='utf-8', errors='ignore') as csvfile:
            for row in islice(csv.reader(csvfile, delimiter=self.delimeter), self.start_point+item*self.batch_size,None):
                self.articles.append([int(row[0]),preprocess_sentence(row[1])])
                if len(self.articles) >= self.batch_size:
                    break

        self.articles = np.array(self.articles)
        if len(self.articles.shape) < 2:
            print("Working around...")
            self.articles = np.array([[0, "fgdssdgdsfgdsfgdsfg"], [1, "fgdssdgdsfgdsfgdsfg"]])


