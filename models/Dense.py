from aliaser import Sequential,Dense
from text_generators.training_text_generator import TrainingTextGenerator
from models.Model import Model

class DenseModel(Model):
    def __init__(self):
        super().__init__()
        self.model = Sequential()
        self.model_name = "Dense-"
        self.train_text_generator = TrainingTextGenerator
        self.preprocess=True


    def get_uncompiled_model(self):
        if self.topic_nums is None or self.enhanced_num_of_topics is None or self.num_of_words is None:
            raise Exception("Base argument were not set. Network cannot be created.")
        self.model = Sequential()
        self.model.add(Dense(self.enhanced_num_of_topics, activation='relu', input_shape=(self.num_of_words,)))
        self.model.add(Dense(self.enhanced_num_of_topics, activation='relu'))
        self.model.add(Dense(self.topic_nums, activation='softmax'))
        return self.model

    def get_compiled_model(self):
        self.get_uncompiled_model().compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

