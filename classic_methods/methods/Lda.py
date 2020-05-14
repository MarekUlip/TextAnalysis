import os
import gensim
from gensim import corpora
from gensim.test.utils import datapath


class Lda:
    def __init__(self, topic_count=5, topic_word_count=15, alpha="auto", eta="auto", kappa=0.51, tau=2.0, minimum_probability=0.0, passes=25, iterations=25, random_state=5, params=None):
        """
        Class for working wit LDA from gensim library
        :param topic_count: num of topics (or clusters) to be found
        :param topic_word_count: num of topic words to be printed if neccessary
        :param alpha: number of expected topics that expresses our a-priori belief for the each topics’ probability
        :param eta: A-priori belief on word probability
        :param kappa: A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten when each new document is examined.
        :param tau: Hyper-parameter that controls how much we will slow down the first steps the first few iterations
        :param minimum_probability: minimum probability to keep word
        :param passes: Number of passes through the corpus during training.
        :param iterations: Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
        :param random_state: used for testing
        :param params: dictionary containing init parameters. If this var is not none it is prevered over any other init parameter in method signature. Unspecified key value will be set to default value
        """
        if params is not None:
            self.topic_count = params.get("topic_count", topic_count)
            self.topic_word_count = params.get("topic_word_count", topic_word_count)
            self.alpha = params.get("alpha", alpha)
            self.eta = params.get("eta", eta)
            self.kappa = params.get("kappa", kappa)
            self.tau = params.get("tau", tau)
            self.minimum_probability = params.get("minimum_probability", minimum_probability)
            self.passes = params.get("passes", passes)
            self.iterations = params.get("iterations", iterations)
            self.random_state = params.get("random_state", random_state)
        else:
            self.topic_count = topic_count
            self.topic_word_count = topic_word_count
            self.alpha = alpha
            self.eta = eta
            self.kappa = kappa
            self.tau = tau
            self.minimum_probability = minimum_probability
            self.passes = passes
            self.iterations = iterations
            self.random_state = random_state
        self.dictionary = None
        self.model = None
        self.model_folder = os.getcwd()+"\\lda\\"
        self.model_path = self.model_folder+"model"
        self.dictionary_path = self.model_folder+"dictionary"

    def train(self, texts):
        """
        Trains this model with provided texts
        :param texts: list of tuples in form (topic_id, text) topic ids dont matter here this format is used only because
        its more general - easier testing
        """
        texts = [text[1].split() for text in texts]
        self.dictionary = corpora.Dictionary(texts)
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in texts]
        self.model = gensim.models.LdaModel(
            doc_term_matrix,
            num_topics=self.topic_count,
            id2word=self.dictionary,
            alpha=self.alpha,
            # eta=int(self.eta.get()),
            decay=self.kappa,
            offset=self.tau,
            # random_state=5,
            passes=self.passes,
            iterations=self.iterations)

    def extract_important_words(self, topics, keep_values=True):
        """
        Methot to obtaion topic words from this model. Prefered alternative is print_topics or show_topics from model variable
        :param topics: topics from this model
        :param keep_values: whether to keep numeric values of each topic word
        :return: list of topic words (or tuples if keep values is set to true)
        """
        d = {}
        i = 0
        for x in topics:
            a = x[1].replace(" ", "")
            a = a.replace("\"", "")
            d[i] = []
            for y in a.split("+"):
                if keep_values:
                    d[i].append(tuple(y.split("*")))
                else:
                    d[i].append(y.split("*")[1])
            i += 1
        return d

    def analyse_text(self, text):
        """
        Analyses provided text and returns index of most significant topic
        :param text: text to be analysed in form of string
        :return: index number of most significant topic
        """
        bow = self.dictionary.doc2bow(text.split())
        return self.model[bow]

    def get_topics(self):
        """
        Get model topics with their words base on topic_word_count parameter.
        :return: model topics with their words.
        """
        return self.model.print_topics(-1, self.topic_word_count)

