from keras.utils.np_utils import to_categorical
import numpy as np
from keras.preprocessing.text import Tokenizer
from text_generators.text_generator import TextGenerator

class TrainingTextGeneratorRNN(TextGenerator):
    def __init__(self, filename, batch_size, num_of_texts, num_of_words, tokenizer: Tokenizer, delimeter,
                 num_of_classes, max_len=None, start_point=0, preprocess=False):
        super().__init__(filename, batch_size, num_of_texts, num_of_words, tokenizer, delimeter, num_of_classes,
                         max_len, start_point, preprocess)

    def __getitem__(self, item):
        super().__getitem__(item)
        labels = to_categorical(self.articles[:,0], num_classes=self.num_of_classes, dtype=np.uint8)
        features = self.tokenizer.texts_to_matrix(self.articles[:,1],mode="binary")
        self.articles = None
        return np.reshape(features,(features.shape[0], 1, features.shape[1])), labels