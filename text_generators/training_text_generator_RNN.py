from neural_networks.aliaser import Tokenizer
import numpy as np
from text_generators.text_generator import TextGenerator

class TrainingTextGeneratorRNN(TextGenerator):
    def __init__(self, filename, batch_size, num_of_texts, num_of_words, tokenizer: Tokenizer, delimeter,
                 dataset_helper, max_len=None, start_point=0, preprocess=False,preload_dataset=True, is_predicting=False, tokenizer_mode='binary'):
        super().__init__(filename, batch_size, num_of_texts, num_of_words, tokenizer, delimeter, dataset_helper,
                         max_len, start_point, preprocess,preload_dataset,is_predicting, tokenizer_mode)

    def __getitem__(self, item):
        super().__getitem__(item)
        labels = self.get_labels()#to_categorical(self.tmp_articles[:,0], num_classes=self.num_of_classes, dtype=np.uint8)
        features = self.get_basic_tokenizer_matrix()
        self.tmp_articles = None
        if self.is_predicting:
            return np.reshape(features,(features.shape[0], 1, features.shape[1]))
        else:
            return np.reshape(features,(features.shape[0], 1, features.shape[1])), labels