from aliaser import Tokenizer
import numpy as np
class NeuralTopicMatrix:
    def __init__(self, matrix, word_mappings, num_of_topics=None, tokenizer=None):
        self.matrix = matrix
        if num_of_topics is None:
            self.num_of_topics = len(matrix)
        self.tokenizer: Tokenizer = tokenizer
        self.num_of_topics = num_of_topics
        self.word_mappings = word_mappings

    def analyse_text(self, document):
        if self.tokenizer is not None:
            doc = self.tokenizer.texts_to_matrix([document])
            groups = []
            for i in range(self.num_of_topics):
                groups.append(np.sum(np.multiply(doc,self.matrix[i])))
            return [[np.argwhere(groups==np.max(groups))[0][0],0]]
        if type(document) is not list:
            document = document.split()

        return [[]]

    def analyse_documents(self, documents):
        pass
