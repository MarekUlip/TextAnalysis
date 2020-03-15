from neural_networks.aliaser import Sequential,LSTM,Dense,Embedding,Dropout
from text_generators.training_text_generator_RNN_embedding import TrainingTextGeneratorRNNEmbedding
from neural_networks.models.Model import Model
from neural_networks.embedding_loader import get_embedding_matrix

class EmbeddingLSTMModel(Model):
    def __init__(self):
        super().__init__()
        self.model = Sequential()
        self.model_name = "LSTM-Embedding-"
        self.activation_functions = 'tanh'
        self.train_text_generator = TrainingTextGeneratorRNNEmbedding
        self.embedding_matrix = []
        self.embedding_dim = 100
        self.preprocess=True
        self.tokenizer = None


    def get_uncompiled_static_model(self):
        if self.topic_nums is None or self.enhanced_num_of_topics is None or self.num_of_words is None:
            raise Exception("Base argument were not set. Network cannot be created.")
        self.model = Sequential()
        self.model.add(Embedding(self.num_of_words,self.embedding_dim))
        self.model.add(LSTM(self.enhanced_num_of_topics, input_shape=(1, self.num_of_words), return_sequences=True))
        self.model.add(LSTM(self.enhanced_num_of_topics))
        self.model.add(Dense(self.topic_nums, activation='softmax'))

        self.model.layers[0].set_weights([self.embedding_matrix])
        self.model.layers[0].trainable = False
        return self.model

    def get_compiled_static_model(self):
        self.get_uncompiled_static_model().compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def get_uncompiled_model(self):
        self.correct_params()
        last_lay_num = self.num_of_layers-1
        self.model = Sequential()
        self.model.add(Embedding(self.num_of_words,self.embedding_dim))
        if self.num_of_layers != 1:
            self.model.add(LSTM(self.num_of_neurons[0],return_sequences=True,activation=self.activation_functions[0]))

        for i in range(1,last_lay_num):
            if self.dropouts[i]:
                self.model.add(Dropout(rate=self.dropout_values[i]))
            self.model.add(LSTM(self.num_of_neurons[i],return_sequences=True,activation=self.activation_functions[i]))
        self.model.add(LSTM(self.num_of_neurons[last_lay_num], activation=self.activation_functions[last_lay_num]))
        self.model.add(Dense(self.topic_nums,activation='softmax'))


        self.model.layers[0].set_weights([get_embedding_matrix(self.num_of_words, self.embedding_dim, self.tokenizer.word_index)])
        self.model.layers[0].trainable = False
        return self.model

    def get_compiled_model(self):
        self.get_uncompiled_model().compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model


