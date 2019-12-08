from aliaser import Sequential, GRU, Dense,Dropout
from text_generators.training_text_generator_RNN import TrainingTextGeneratorRNN
from models.Model import Model

class GRUModel(Model):
    def __init__(self):
        super().__init__()
        self.model = Sequential()
        self.model_name = "GRU-"
        self.activation_functions = 'tanh'
        self.train_text_generator = TrainingTextGeneratorRNN
        self.preprocess=True


    def get_uncompiled_static_model(self):
        if self.topic_nums is None or self.enhanced_num_of_topics is None or self.num_of_words is None:
            raise Exception("Base argument were not set. Network cannot be created.")
        self.model = Sequential()
        self.model.add(GRU(self.enhanced_num_of_topics, input_shape=(1, self.num_of_words), return_sequences=True))
        self.model.add(GRU(self.enhanced_num_of_topics))
        self.model.add(Dense(self.topic_nums, activation='softmax'))
        return self.model

    def get_compiled_static_model(self):
        self.get_uncompiled_model().compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def get_uncompiled_model(self):
        self.correct_params()
        last_lay_num = self.num_of_layers-1
        self.model = Sequential()
        if self.num_of_layers == 1:
            self.model.add(GRU(self.num_of_neurons[0],input_shape=(1, self.num_of_words),activation=self.activation_functions[0]))
        else:
            self.model.add(GRU(self.num_of_neurons[0],input_shape=(1, self.num_of_words),return_sequences=True,activation=self.activation_functions[0]))
        for i in range(1,last_lay_num):
            if self.dropouts[i]:
                self.model.add(Dropout(rate=self.dropout_values[i]))
            self.model.add(GRU(self.num_of_neurons[i],return_sequences=True,activation=self.activation_functions[i]))
        if self.num_of_layers != 1:
            self.model.add(GRU(self.num_of_neurons[last_lay_num], activation=self.activation_functions[last_lay_num]))
        #self.model.add(GRU(self.num_of_neurons[last_lay_num], activation=self.activation_functions[last_lay_num]))
        self.model.add(Dense(self.topic_nums,activation='softmax'))
        return self.model

    def get_compiled_model(self):
        self.get_uncompiled_model().compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

