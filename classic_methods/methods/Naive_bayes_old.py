from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB




class NaiveBayes:
    def __init__(self):
        self.model = None
        self.articles = []
        self.tfidf = None
        self.test_start_index = 0
        self.end = 0


    def train(self, texts_for_train, texts_for_test):
        """
        Trains the model with provided texts. I also prepares for analysis of unseen text (Naive bayes reqiures to know
        all words so tfidf matrix must be created from all texts (train and test)
        :param texts_for_train: list of tuples in form of (topic id, text) used for model training
        :param texts_for_test: list of tuples in form of (topic id, text) used for analyisis or testing
        """
        topics = []
        self.test_start_index = len(texts_for_train)
        texts = texts_for_train.copy()
        texts.extend(texts_for_test) #TODO careful for rewrites
        for text in texts:
            self.articles.append(text[1])
            topics.append(text[0])

        counts = CountVectorizer().fit_transform(self.articles)
        self.tfidf = TfidfTransformer().fit_transform(counts)
        self.model = MultinomialNB().fit(self.tfidf[0:self.test_start_index], topics[0:self.test_start_index])
        self.end = len(self.articles)

    def analyse_texts(self):
        """
        Analyses texts for test which are provided via constructor
        :return: list of topic indexes for each document contained in texts for tests
        """
        return self.model.predict(self.tfidf[self.test_start_index:self.end])

    def get_topics(self):
        """
        This model does not return topics. (Functuion was kept for compatability reasons)
        :return: list
        """
        return ["Naive bayes only knows indexes. Topic words ommited."]