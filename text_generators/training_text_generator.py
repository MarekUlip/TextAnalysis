from aliaser import to_categorical, Tokenizer
import numpy as np
from text_generators.text_generator import TextGenerator


class TrainingTextGenerator(TextGenerator):

    def __init__(self, filename, batch_size, num_of_texts, num_of_words, tokenizer: Tokenizer, delimeter,
                 dataset_helper, max_len=None, start_point=0, preprocess=False,preload_dataset=True, is_predicting=False):
        super().__init__(filename, batch_size, num_of_texts, num_of_words, tokenizer, delimeter, dataset_helper,
                         max_len, start_point, preprocess,preload_dataset,is_predicting)

    def get_dataset(self):
        super().get_dataset()
        labels = to_categorical(self.tmp_articles[:, 0], num_classes=self.num_of_classes, dtype=np.uint8)
        features = self.tokenizer.texts_to_matrix(self.tmp_articles[:, 1], mode="binary")
        self.tmp_articles = None
        if self.is_predicting:
            return features
        else:
            return features, labels

    def __getitem__(self, item):
        super().__getitem__(item)
        labels = to_categorical(self.tmp_articles[:, 0], num_classes=self.num_of_classes, dtype=np.uint8)
        features = self.tokenizer.texts_to_matrix(self.tmp_articles[:, 1], mode="binary")
        self.tmp_articles = None
        if self.is_predicting:
            return features
        else:
            return features, labels
