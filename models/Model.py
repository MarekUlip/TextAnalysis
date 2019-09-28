import numpy as np


class Model:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.train_text_generator = None
        self.topic_nums = None
        self.enhanced_num_of_topics = None
        self.num_of_words = None
        self.preprocess = None
        self.max_len = None
        pass

    def requires_preprocess(self):
        return self.preprocess

    def set_base_params(self, topic_nums, num_of_words):
        self.topic_nums = topic_nums
        self.enhanced_num_of_topics = int(np.ceil(topic_nums) * 2.5)
        self.num_of_words = num_of_words

    def get_description(self):
        return self.model_name

    def get_uncompiled_model(self):
        pass

    def get_compiled_model(self):
        pass

    def get_current_model(self):
        return self.model

    def fit_generator(self, datasets_helper, batch_size, tokenizer, validation_count, epochs=2, max_len=None):
        return self.model.fit_generator(
            generator=self.train_text_generator(datasets_helper.get_train_file_path(), batch_size,
                                                datasets_helper.get_num_of_train_texts(),
                                                self.num_of_words, tokenizer, ";",
                                                datasets_helper.get_num_of_topics(),max_len=max_len,preprocess=self.preprocess), epochs=epochs,
            validation_data=self.train_text_generator(datasets_helper.get_train_file_path(),
                                                      batch_size, validation_count, self.num_of_words,
                                                      tokenizer, ";",
                                                      datasets_helper.get_num_of_topics(),
                                                      start_point=datasets_helper.get_num_of_train_texts() - validation_count,max_len=max_len,preprocess=self.preprocess))

    def evaluate_generator(self, datasets_helper, batch_size, tokenizer):
        return self.model.evaluate_generator(
            generator=self.train_text_generator(datasets_helper.get_test_file_path(), batch_size,
                                                datasets_helper.get_num_of_test_texts(),
                                                self.num_of_words, tokenizer, ";",
                                                datasets_helper.get_num_of_topics(),max_len=self.max_len, preprocess=self.preprocess))

