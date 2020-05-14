import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import numpy as np

class LdaSklearn:
    def __init__(self, topic_count=5, passes=20, iterations=20, params=None):
        """
        Class for work with LDA implementation of sklearn library
        :param topic_count: number of topics (clusters) to be considered during training
        :param passes:
        :param iterations:
        :param params: parameters for this model represented with dictionary. None specified values will be converted into default values. keys {topic_count, passes, iterations, max_feauters)
        """
        if params is not None:
            self.topic_count = params.get("topic_count", 3)
            self.passes = params.get("passes", passes)
            self.iterations = params.get("iterations", iterations)
        else:
            self.topic_count = topic_count
            self.passes = passes
            self.iterations = iterations
        self.model = None
        self.max_features = params.get('max_features',10000)
        self.tf_vectorizer = TfidfVectorizer(max_features=self.max_features)
        #paths in case this model would be saved
        self.model_folder = os.getcwd()+"\\lda-sklearn\\"
        self.model_path = self.model_folder+"model"
        self.dictionary_path = self.model_folder+"dictionary"

    def train(self, texts):
        """
        Trains this model with provided texts
        :param texts: list of tuples in form (topic_id, text) topic ids dont matter here this format is used only because
        its more general - easier testing
        """
        texts = [text[1] for text in texts]
        train = self.tf_vectorizer.fit_transform(texts)
        self.model = LatentDirichletAllocation(n_components=self.topic_count, max_iter=self.iterations, learning_offset=2.0, learning_decay=0.51)
        self.model.n_iter_ = self.passes
        self.model.fit(train)

    def extract_important_words(self, topics, keep_values=True):
        """
        Not implemented for this model kept back for compatability reasons
        """
        pass

    def analyse_text(self, text):
        """
        Analyses provided text and returns list with topic index of most possible topic
        :param text: text to be analysed
        :return: list with topic index of most possible topic
        """
        #Returning list of one int is for testing compatability
        test = self.tf_vectorizer.transform(["".join(word for word in text)])
        topic_dist = np.matrix(self.model.transform(test))
        return [(topic_dist.argmax(axis=1).item(0), 1)]

    def get_topics(self):
        """
        Not implemented kept back for compatability reasons
        :return: empty list
        """
        return []#self.model.print_topics(-1, self.topic_word_count)

