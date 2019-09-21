from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import csv
import numpy as np
from itertools import islice


class Simple_Text_Generator(Sequence):
    def __init__(self, filename, batch_size, num_of_texts, delimeter):
        self.filename = filename
        self.batch_size = batch_size
        self.num_of_texts = num_of_texts
        self.delimeter = delimeter

    def __len__(self):
        return np.ceil(self.num_of_texts/self.batch_size).astype(np.int)

    def __getitem__(self, item):
        print("returning batch for {} item".format(item))
        articles = []
        with open(self.filename, encoding='utf-8', errors='ignore') as csvfile:
            for row in islice(csv.reader(csvfile, delimiter=self.delimeter), item*self.batch_size, None):
                articles.append(row[1])
                if len(articles) >= self.batch_size:
                    break

        return articles