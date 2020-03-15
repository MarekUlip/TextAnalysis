from neural_networks.aliaser import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
class NeuralTopicMatrix:
    def __init__(self, matrix, word_mappings, num_of_topics=None, tokenizer=None):
        self.matrix = matrix
        if num_of_topics is None:
            self.num_of_topics = len(matrix)
        self.tokenizer: Tokenizer = tokenizer
        self.num_of_topics = num_of_topics
        self.word_mappings = word_mappings

    def analyse_text(self, document, return_in_gensim_style=False):
        if self.tokenizer is not None:
            doc = self.tokenizer.texts_to_matrix([document])
            groups = []
            for i in range(self.num_of_topics):
                if return_in_gensim_style:
                    groups.append((i,np.sum(np.multiply(doc,self.matrix[i]))))
                else:
                    groups.append(np.sum(np.multiply(doc,self.matrix[i])))
            if return_in_gensim_style:
                return groups
            else:
                return [[np.argwhere(groups==np.max(groups))[0][0],0]]
        if type(document) is not list:
            document = document.split()

        return [[]]

    def analyse_documents(self, documents):
        pass

class NeuralTopicMatrixTFIDF:
    def __init__(self, matrix, word_mappings, num_of_topics=None, tokenizer=None):
        self.matrix = matrix
        if num_of_topics is None:
            self.num_of_topics = len(matrix)
        self.tokenizer: TfidfVectorizer = tokenizer
        self.num_of_topics = num_of_topics
        self.word_mappings = word_mappings

    def analyse_text(self, document, return_in_gensim_style=False):
        if self.tokenizer is not None:
            doc = self.tokenizer.transform([document]).todense()
            groups = []
            for i in range(self.num_of_topics):
                if return_in_gensim_style:
                    groups.append((i,np.sum(np.multiply(doc,self.matrix[i]))))
                else:
                    groups.append(np.sum(np.multiply(doc,self.matrix[i])))
            if return_in_gensim_style:
                return groups
            else:
                return [[np.argwhere(groups==np.max(groups))[0][0],0]]
        if type(document) is not list:
            document = document.split()

        return [[]]

    def analyse_documents(self, documents):
        pass