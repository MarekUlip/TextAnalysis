from aliaser import Sequential, GRU, Dense
from text_generators.training_text_generator import TrainingTextGenerator
from models.Model import Model

class GRUModel(Model):
    def __init__(self):
        super().__init__()
        self.model = Sequential()
        self.model_name = "GRU-"
        self.train_text_generator = TrainingTextGenerator
        self.preprocess=True


    def get_uncompiled_model(self):
        if self.topic_nums is None or self.enhanced_num_of_topics is None or self.num_of_words is None:
            raise Exception("Base argument were not set. Network cannot be created.")
        self.model = Sequential()
        self.model.add(GRU(self.enhanced_num_of_topics, input_shape=(1, self.num_of_words), return_sequences=True))
        self.model.add(GRU(self.enhanced_num_of_topics))
        self.model.add(Dense(self.topic_nums, activation='softmax'))
        return self.model

    def get_compiled_model(self):
        self.get_uncompiled_model().compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

