from aliaser import to_categorical, Tokenizer
import numpy as np
from text_generators.text_generator import TextGenerator


class TrainingTextGenerator(TextGenerator):

    def __init__(self, filename, batch_size, num_of_texts, num_of_words, tokenizer: Tokenizer, delimeter,
                 num_of_classes, max_len=None, start_point=0, preprocess=False):
        super().__init__(filename, batch_size, num_of_texts, num_of_words, tokenizer, delimeter, num_of_classes,
                         max_len, start_point, preprocess)

    def __getitem__(self, item):
        super().__getitem__(item)
        labels = to_categorical(self.articles[:, 0], num_classes=self.num_of_classes, dtype=np.uint8)
        features = self.tokenizer.texts_to_matrix(self.articles[:, 1], mode="binary")
        self.articles = None
        return features, labels
