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
        self.num_of_layers = None
        self.num_of_neurons:list = None
        self.activation_functions = None
        self.optimizer = 'rmsprop'
        pass

    def requires_preprocess(self):
        return self.preprocess

    def set_base_params(self, topic_nums, num_of_words):
        self.topic_nums = topic_nums
        self.enhanced_num_of_topics = int(np.ceil(topic_nums) * 2.5)
        self.num_of_words = num_of_words

    def create_list_from_value(self,value,n):
        """
        Creates list of n numbers with specified value. Function is inteded to be used when number of neurons in all neurons
        is supposed to be same.
        :param value: Number to be listified
        :param n: number of times the number should be listified
        :return: list of n numbers of specified value
        """
        return [value for _ in range(n)]

    def set_params(self, params:dict):
        """
        Sets provided parameters for this model. Unspecified params will be unchanged. Their default value is None.
        :param params: dictionary with params to be changed. Keys should match class variables
        :return:
        """
        if params.get('model',None) is not None:
            self.model = params['model']
        if params.get('model_name',None) is not None:
            self.model_name = params['model_name']
        if params.get('train_text_generator',None) is not None:
            self.train_text_generator = params['train_text_generator']
        if params.get('topic_nums',None) is not None:
            self.topic_nums = params['topic_nums']
        if params.get('enhanced_num_of_topics',None) is not None:
            self.enhanced_num_of_topics = params['enhanced_num_of_topics']
        if params.get('num_of_words',None) is not None:
            self.num_of_words = params['num_of_words']
        if params.get('preprocess',None) is not None:
            self.preprocess = params['preprocess']
        if params.get('max_len',None) is not None:
            self.max_len = params['max_len']
        if params.get('num_of_layers',None) is not None:
            self.num_of_layers = params['num_of_layers']
        if params.get('num_of_neurons',None) is not None:
            self.num_of_neurons = params['num_of_neurons']
        if params.get('activation_functions',None) is not None:
            self.activation_functions = params['activation_functions']
        if params.get('optimizer',None) is not None:
            self.optimizer = params['optimizer']

    def correct_params(self):
        if type(self.num_of_neurons) is not list:
            self.num_of_neurons = self.create_list_from_value(self.num_of_neurons,self.num_of_layers)
        else:
            self.num_of_layers = len(self.num_of_neurons)
        if type(self.activation_functions) is not list:
            self.activation_functions = self.create_list_from_value(self.activation_functions,self.num_of_neurons)

    def get_description(self):
        return self.model_name

    def get_uncompiled_model(self):
        pass

    def get_compiled_model(self):
        pass

    def get_compiled_static_model(self):
        pass

    def get_uncompiled_static_model(self):
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

