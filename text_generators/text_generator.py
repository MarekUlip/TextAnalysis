from neural_networks.aliaser import Sequence, Tokenizer, to_categorical
import numpy as np
from dataset_loader.dataset_helper import Dataset_Helper

class TextGenerator(Sequence):
    def __init__(self, filename, batch_size, num_of_texts, num_of_words, tokenizer: Tokenizer, delimeter,
                 dataset_helper:Dataset_Helper, max_len=None, start_point=0, preprocess=False,preload_dataset=True, is_predicting=False, tokenizer_mode='binary'):
        self.filename = filename
        self.batch_size = batch_size
        self.num_of_texts = num_of_texts
        self.tokenizer = tokenizer
        self.delimeter = delimeter
        self.num_of_words = num_of_words
        self.num_of_classes = dataset_helper.get_num_of_topics() #num_of_classes
        self.start_point = start_point
        self.max_len = max_len
        self.preprocess = preprocess
        self.preload_dataset = preload_dataset
        self.is_predicting = is_predicting
        self.dataset_helper = dataset_helper
        self.tokenizer_mode = tokenizer_mode
        self.labels = []
        self.tmp_articles = None
        self.articles = []
        if preload_dataset:
            self.load_dataset()

    def __len__(self):
        return np.ceil(self.num_of_texts / self.batch_size).astype(np.int)

    def load_dataset(self):
        """with open(self.filename, encoding='utf-8', errors='ignore') as csvfile:
            for row in csv.reader(csvfile, delimiter=self.delimeter):
                self.articles.append([int(row[0]),preprocess_sentence(row[1]) if self.preprocess else row[1]])"""
        self.articles = self.dataset_helper.get_dataset(path=self.filename)

    def get_dataset(self):
        self.tmp_articles = np.array(self.articles)
        if self.is_predicting:
            self.labels.extend(list(map(int,self.tmp_articles[:,0])))

    def get_labels(self):
        if self.dataset_helper.vectorized_labels:
            return self.tmp_articles[:,0]
        else:
            return to_categorical(self.tmp_articles[:,0], num_classes=self.num_of_classes, dtype=np.uint8)

    def get_basic_tokenizer_matrix(self):
        return self.tokenizer.texts_to_matrix(self.tmp_articles[:, 1], mode=self.tokenizer_mode)

    def __getitem__(self, item):
        self.tmp_articles = []
        if self.preload_dataset:
            self.tmp_articles = self.articles[self.start_point+item*self.batch_size:self.start_point+(item+1)*self.batch_size]
        else:
            """with open(self.filename, encoding='utf-8', errors='ignore') as csvfile:
                for row in islice(csv.reader(csvfile, delimiter=self.delimeter), self.start_point+item*self.batch_size,None):
                    self.tmp_articles.append([int(row[0]),preprocess_sentence(row[1])])
                    if len(self.tmp_articles) >= self.batch_size:
                        break"""
            self.tmp_articles = self.dataset_helper.get_dataset_slice(self.start_point+item*self.batch_size,self.batch_size,path=self.filename)

        self.tmp_articles = np.array(self.tmp_articles)
        if len(self.tmp_articles.shape) < 2:
            print("Working around...")
            print(self.start_point)
            self.tmp_articles = np.array([[0, "fgdssdgdsfgdsfgdsfg"]])
        if self.is_predicting:
            self.labels.extend(list(map(int,self.tmp_articles[:,0])))


