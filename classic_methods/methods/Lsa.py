import os
import gensim
from gensim import corpora, similarities
from gensim.test.utils import datapath

class Lsa:
    def __init__(self, topic_count=10, topic_word_count=15, one_pass=False, power_iter=2, extra_samples=1, decay=1.0, use_tfidf=True, params=None):
        """
        Creates untrained LSA model with specified parameters.
        :param topic_count: number of topics that will be encountered in dataset
        :param topic_word_count: number of words that will be printed with get_topics method
        :param use_tfidf: Indicates whether lsa should work with tfidf matrix
        :param params: parameters for this model represented with dictionary. None specified values will be converted into default values.
        """
        if params is not None:
            self.topic_count = params.get("topic_count", topic_count)
            self.topic_word_count = params.get("topic_word_count", topic_word_count)
            self.one_pass = params.get("one_pass", one_pass)
            self.power_iter = params.get("power_iter", power_iter)
            self.extra_samples = params.get("extra_samples", extra_samples)
            self.decay = params.get("decay", decay)
            self.use_tfidf = params.get("use_tfidf", use_tfidf)
        else:
            self.topic_count = topic_count
            self.topic_word_count = topic_word_count
            self.one_pass = one_pass
            self.power_iter = power_iter
            self.extra_samples = extra_samples
            self.decay = decay
            self.use_tfidf = use_tfidf
        self.dictionary = None
        self.model = None
        self.index = None
        self.corpus = None
        self.topics = []


    def train(self, texts_in):
        """
        Trains this model with provided texts
        :param texts: list of tuples in form (topic_id, text) topic ids dont matter here this format is used only because
        its more general - easier testing
        """
        texts = []
        for text in texts_in:
            self.topics.append(int(text[0]))
            texts.append(text[1].split())
        #self.topics = [int(text[0]) for text in texts]
        #texts = [text[1].split() for text in texts]
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in texts]
        if self.use_tfidf:
            tf_idf = gensim.models.TfidfModel(self.corpus)
            corpus_tfidf = tf_idf[self.corpus]
        else:
            corpus_tfidf = self.corpus
        self.model = gensim.models.LsiModel(corpus_tfidf,
                                            id2word=self.dictionary,
                                            num_topics=self.topic_count,
                                            onepass=self.one_pass,
                                            power_iters=self.power_iter,
                                            decay=self.decay)
        self.index = similarities.MatrixSimilarity(self.model[self.corpus])


        #return lsamodel.print_topics(num_topics=self.topic_count, num_words=self.topic_word_count)

    def analyse_text(self, text):
        """
        Analyses provided text and returns topic index of the most possible topic
        :param text:
        :return: integer index of topic
        """
        vec_bow = self.dictionary.doc2bow(text.split())
        vec_lsi = self.model[vec_bow]
        sims = self.index[vec_lsi]
        #simsT = sorted(enumerate(sims), key=lambda item: -item[1])
        sim = max(enumerate(sims), key=lambda item: item[1])
        #sim = simsT[0]
        return self.topics[sim[0]]#[self.topics[sim[0]], sim[1]]#self.model[bow]

    def get_topics(self):
        """
        :return: words assosiated with each topic
        """
        return self.model.print_topics(-1, self.topic_word_count)