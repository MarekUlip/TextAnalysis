from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC




class SupportVectorMachines:
    def __init__(self, params:dict=None):
        """
        Class working with SVM implementation from sklearn library.
        :param params: dict containing init values. Accepted keys are C, kernel, degree, gamma and max_features. Check documentation of sklearn SVC for meaning of those variables.
        """
        if params is not None:
            self.model = SVC(C=100, kernel='rbf', degree=3, gamma=1)
        else:
            self.model = SVC(C=params.get('c',100),kernel=params.get('kernel','rbf'),degree=params.get('degree',3),gamma=params.get('gamma',1))
        self.articles = []
        self.max_features = params.get('max_features',10000)
        self.tfidf = None
        self.test_start_index = 0
        self.end = 0


    def train(self, texts_for_train):
        """
        Trains the model with provided texts. I also prepares for analysis of unseen text (Naive bayes reqiures to know
        all words so tfidf matrix must be created from all texts (train and test)
        :param texts_for_train: list of tuples in form of (topic id, text) used for model training
        :param texts_for_test: list of tuples in form of (topic id, text) used for analyisis or testing
        """
        topics = []
        self.test_start_index = len(texts_for_train)
        texts = texts_for_train.copy()
        #texts.extend(texts_for_test) #TODO careful for rewrites
        for text in texts:
            self.articles.append(text[1])
            topics.append(text[0])


        self.tfidf = TfidfVectorizer(max_features=self.max_features)
        self.model = self.model.fit(self.tfidf.fit_transform(self.articles), topics)
        self.end = len(self.articles)

    def analyse_texts(self, texts):
        """
        Analyses texts for test which are provided via constructor
        :return: list of topic indexes for each document contained in texts for tests
        """
        articles = []
        for text in texts:
            articles.append(text[1])
        tfidf = self.tfidf.transform(articles)
        return self.model.predict(tfidf)

    def get_topics(self):
        """
        This model does not return topics. (Functuion was kept for compatability reasons)
        :return: list
        """
        return ["Naive bayes only knows indexes. Topic words ommited."]