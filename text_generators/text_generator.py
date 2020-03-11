from aliaser import Sequence, Tokenizer
import numpy as np
import csv
from itertools import islice
from dataset_helper import preprocess_sentence

class TextGenerator(Sequence):
    def __init__(self, filename, batch_size, num_of_texts, num_of_words, tokenizer: Tokenizer, delimeter,
                 num_of_classes, max_len=None, start_point=0, preprocess=False,preload_dataset=True, is_predicting=False):
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
        self.preload_dataset = preload_dataset
        self.is_predicting = is_predicting
        self.labels = []
        self.tmp_articles = None
        self.articles = []
        if preload_dataset:
            self.load_dataset()

    def __len__(self):
        return np.ceil(self.num_of_texts / self.batch_size).astype(np.int)

    def load_dataset(self):
        with open(self.filename, encoding='utf-8', errors='ignore') as csvfile:
            for row in csv.reader(csvfile, delimiter=self.delimeter):
                self.articles.append([int(row[0]),preprocess_sentence(row[1]) if self.preprocess else row[1]])

    def get_dataset(self):
        self.tmp_articles = np.array(self.articles)
        if self.is_predicting:
            self.labels.extend(list(map(int,self.tmp_articles[:,0])))


    def __getitem__(self, item):
        self.tmp_articles = []
        if self.preload_dataset:
            self.tmp_articles = self.articles[self.start_point+item*self.batch_size:self.start_point+(item+1)*self.batch_size]
        else:
            with open(self.filename, encoding='utf-8', errors='ignore') as csvfile:
                for row in islice(csv.reader(csvfile, delimiter=self.delimeter), self.start_point+item*self.batch_size,None):
                    self.tmp_articles.append([int(row[0]),preprocess_sentence(row[1])])
                    if len(self.tmp_articles) >= self.batch_size:
                        break

        self.tmp_articles = np.array(self.tmp_articles)
        if len(self.tmp_articles.shape) < 2:
            print("Working around...")
            self.tmp_articles = np.array([[0, "fgdssdgdsfgdsfgdsfg"]])
        if self.is_predicting:
            self.labels.extend(list(map(int,self.tmp_articles[:,0])))


