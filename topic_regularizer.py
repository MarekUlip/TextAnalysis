"""Built-in regularizers.
"""

import six
from gensim.models import CoherenceModel
from tensorflow.keras import backend as K
from tensorflow.keras.utils import serialize_keras_object
from tensorflow.keras.utils import deserialize_keras_object
import numpy as np
import tensorflow as tf


class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TopicRegularizer(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, dictionary=None, corpus=None, num_of_topics=None, num_of_important_words=None, reverse_word_map=None):
        self.dictionary = dictionary
        self.corpus = corpus
        self.num_of_topics = num_of_topics
        self.num_of_important_words = num_of_important_words
        self.reverse_word_map = reverse_word_map

    def __call__(self, x):
        tst = x.numpy()
        #c_v, u_mass = self.measureCoherence(self.get_extremes(x))
        return K.sum(x)#(1-c_v) * np.abs(u_mass)

    def get_config(self):
        return {'l1': 0,
                'l2': 0}

    def get_extremes(self, matrix):
        topic_words = [[] for i in range(self.num_of_topics)]
        for i in range(self.num_of_topics):
            extreme = (-tf.sort(-matrix))[self.num_of_important_words]
            indexes = np.argwhere(matrix[i] >= extreme)
            for index in indexes:
                index = index[0]
                if index == 0:
                    continue
                topic_words[i].append(self.reverse_word_map[index])#[matrix[i, index], reverse_word_map[index]])
        #topic_words = [sorted(words, key=lambda x: x[1], reverse=True) for words in topic_words]

        return topic_words

    def measureCoherence(self, topic_words):
        #topic_words = extract_words(topic_words)
        coherence = CoherenceModel(topics=topic_words, texts=self.corpus, dictionary=self.dictionary, coherence='c_v',
                                   processes=1)
        u_mass = CoherenceModel(topics=topic_words, texts=self.corpus, dictionary=self.dictionary, coherence='u_mass',
                                processes=1)

        return coherence.get_coherence(), u_mass.get_coherence()


# Aliases.


def top_reg(dictionary=None, corpus=None, num_of_topics=None, num_of_important_words=None, reverse_word_map=None):
    return TopicRegularizer(dictionary,corpus,num_of_topics,num_of_important_words,reverse_word_map)


def serialize(regularizer):
    return serialize_keras_object(regularizer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='regularizer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret regularizer identifier: ' +
                         str(identifier))