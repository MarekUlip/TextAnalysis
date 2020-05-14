#from __future__ import absolute_import, division, print_function, unicode_literals

from neural_networks.NeuralLDAanalysisMethods import *
from dataset_loader.dataset_helper import Dataset_Helper
from results_saver import LogWriter
from neural_networks.aliaser import *
import os
import sys
import numpy as np
from neural_networks.lda_impl import Lda
import tkinter as tk

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
root = tk.Tk()
root.withdraw()
from numpy.random import seed
seed(42)
tf.random.set_seed(42)

params = [['LDA-CSFD-simple',14]]
"""params = [['LDA-Yelp',6],
    ['LDA-Reuters',0],
          ['LDA-dbpedia',1],
          ['LDA-20News',3],
          ['LDA-CSFD',12]]"""
for param in params:
    seed(42)
    tf.random.set_seed(42)
    test_name = param[0]
    results = []
    num_of_words = 10000

    dataset_helper = Dataset_Helper(True)
    dataset_helper.set_wanted_datasets([param[1]])
    dataset_helper.next_dataset()
    num_of_topics = dataset_helper.get_num_of_topics()
    documents = dataset_helper.get_texts_as_list()
    labels = dataset_helper.get_labels(dataset_helper.get_train_file_path())
    tokenizer = Tokenizer(num_words=num_of_words)
    tokenizer.fit_on_texts(documents)
    #items= tokenizer.word_index
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    matrix = tokenizer.texts_to_matrix(documents, mode='binary')


    num_of_important_words = 20
    log_writer = LogWriter(log_file_desc='{}{}'.format(test_name,""),result_desc="NeuralTopicModel")

    model = Lda(num_of_topics,num_of_important_words,
                passes=25,
                iterations=25)

    """gensim.models.LdaModel(
    doc_term_matrix,
    num_topics=num_of_topics,
    id2word=dictionary,
    passes=2,
    iterations=2)"""

    #LDA section
    model.train(documents)
    topic_words_lda = extract_important_words(model.get_topics(), True)
    print(topic_words_lda)
    log_writer.write_2D_list('topic_words_lda', topic_words_lda, 'w+')
    test_model(documents, labels, model, log_writer, 'standard_lda')
    #plot_clustering_chart(model,True,documents,log_writer,'lda',dataset_helper.get_dataset_name(),dataset_helper.get_num_of_topics())
    measureCoherence(topic_words_lda, log_writer, model.dictionary, documents, 'lda', dataset_helper.get_dataset_name())

    log_writer.end_logging()




