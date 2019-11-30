from aliaser import Sequential,LSTM,Dense
from text_generators.training_text_generator import TrainingTextGenerator
from models.Model import Model

class LSTMModel(Model):
    def __init__(self):
        super().__init__()
        self.model = Sequential()
        self.model_name = "LSTM-"
        self.activation_functions = 'tanh'
        self.train_text_generator = TrainingTextGenerator
        self.preprocess=True


    def get_uncompiled_static_model(self):
        if self.topic_nums is None or self.enhanced_num_of_topics is None or self.num_of_words is None:
            raise Exception("Base argument were not set. Network cannot be created.")
        self.model = Sequential()
        self.model.add(LSTM(self.enhanced_num_of_topics, input_shape=(1, self.num_of_words), return_sequences=True))
        self.model.add(LSTM(self.enhanced_num_of_topics))
        self.model.add(Dense(self.topic_nums, activation='softmax'))
        return self.model

    def get_compiled_static_model(self):
        self.get_uncompiled_static_model().compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def get_compiled_model(self):
        self.correct_params()
        last_lay_num = self.num_of_layers-1
        self.model = Sequential()
        for i in range(last_lay_num):
            self.model.add(LSTM(self.num_of_neurons[i],input_shape=(1, self.num_of_words),return_sequences=True,activation=self.activation_functions[i]))
        self.model.add(LSTM(self.enhanced_num_of_topics,input_shape=(1, self.num_of_words), activation=self.activation_functions[last_lay_num]))
        self.model.add(Dense(self.topic_nums,activation='softmax'))
        return self.model

    def get_uncompiled_model(self):
        self.get_uncompiled_model().compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model


