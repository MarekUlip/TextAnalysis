#from __future__ import absolute_import, division, print_function, unicode_literals
import operator

from helper_functions import Dataset_Helper
from results_saver import LogWriter
from gensim import corpora
import matplotlib.pyplot as plt
from aliaser import *
import os
import sys
import numpy as np
import gensim
from collections import Counter


def extract_important_words(topics, keep_values=True):
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

def prep_docs_for_assesment(docs, labels):
    """
    Sorts training docs into groups by their respective topics. Used for "gueesing: which model topic index belongs
    to which real topic id.
    :param training_docs: if not provided training docs from initialization will be used otherwise no action
    will be performed
    """
    representants = {}
    for i in range(len(docs)):
        if labels[i] not in representants:
            representants[labels[i]] = [docs[i]]
        else:
            representants[labels[i]].append(docs[i])
    return representants

def connect_topic_id_to_topics(model, representants, log_writer):
    """
    Connects topic indexes from model to topic ids from dataset. Note that some ids might not get connected due various reasons.
    :param model: model containing topics to connect
    """
    # t = model.get_topics()
    topic_indexes = {}
    topics_of_index = {}
    for key, value in representants.items():
        connection_results = {}
        for article in value:
            try:
                # get most possible index
                topic_index = max(model.analyse_text(article), key=lambda item: item[1])[0]
            except ValueError:
                print("No topic index returned continuing")  # TODO replace with if
                continue
            # add most possible index for this article to counter
            if topic_index not in connection_results:
                connection_results[topic_index] = 1
            else:
                connection_results[topic_index] += 1
        # find index that occured mostly
        print(connection_results)
        best_candidates = max(connection_results.items(), key=operator.itemgetter(1))
        print(best_candidates)
        log_writer.add_log(
            "Best candidate with index {} is connected to topic {} with {}% accuracy".format(best_candidates[0],
                                                                                             key, (
                                                                                                         connection_results[
                                                                                                             best_candidates[
                                                                                                                 0]] / len(
                                                                                                     value)) * 100))
        # create connection between topic id and model topic index
        topic_indexes[key] = best_candidates[0]
        # creat connection in opposite direction if there already is some connection add found index to that connection (some model topic index can represent more than one real topic)
        if best_candidates[0] not in topics_of_index:
            topics_of_index[best_candidates[0]] = [key]
        else:
            topics_of_index[best_candidates[0]].append(key)
    return topic_indexes, topics_of_index

def test_model(docs, labels,model, log_writer:LogWriter,test_name):
    """
    Tests provided instance of a model and outputs results using provided test_name
    :param model: model to be tested
    :param test_name: name which will be used for output
    :return: accuracy in range (0 - 1)
    """
    stats = []
    topic_indexes, topics_of_index = connect_topic_id_to_topics(model,prep_docs_for_assesment(docs,labels),log_writer)
    distribution = []
    for index, article in enumerate(docs):
        analysis_res = model.analyse_text(article)
        if len(analysis_res) == 0:
            print("nothing found")
            continue
        res = max(analysis_res, key=lambda item: item[1])
        if res[0] not in topics_of_index:
            topics_of_index[res[0]] = [labels[index]]
            topic_indexes[labels[index]] = res[0]
            print("continuing")
            continue
        distribution.append(res[0])
        stats.append(1 if labels[index] in topics_of_index[res[0]] else 0)
        # self.log_writer.add_log("Article with topic {} was assigned {} with {} certainty.".format(article[0], "correctly" if res[0] == self.topic_positions[article[0]] else "wrong", res[1]))
    accuracy = sum(stats) / len(stats)
    log_writer.add_log("{} got accuracy {}".format(test_name,accuracy))
    log_writer.add_log("Real distribution was {}".format(dict(Counter(labels))))
    log_writer.add_log("Predicted distribution was {}".format(dict(Counter(distribution))))
    return accuracy


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
#sess = tf.compat.v1.Session(config=config)
#tf.keras.backend.set_session(sess)
#results_saver = LogWriter(log_file_desc="Autoencoder")
results = []

num_of_words = 10000
dataset_helper = Dataset_Helper(True)
dataset_helper.set_wanted_datasets([2])
dataset_helper.next_dataset()
num_of_topics = dataset_helper.get_num_of_topics()
documents = dataset_helper.get_texts_as_list()
labels = dataset_helper.get_labels(dataset_helper.get_train_file_path())
tokenizer = Tokenizer(num_words=num_of_words)
tokenizer.fit_on_texts(documents)
#items= tokenizer.word_index
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
matrix = tokenizer.texts_to_matrix(documents, mode='binary')

#mydict = corpora.Dictionary([line.split() for line in documents],prune_at=num_of_words)
#corpus = [mydict.doc2bow(line.split()) for line in documents]

#tfidf = TfidfModel(corpus)
#print(tfidf)

"""model = Sequential()
model.add(Dense(num_of_words*num_of_topics,activation='relu', input_shape=(num_of_words,)))
model.add(Dense(num_of_words,activation='sigmoid'))"""
regularization = 0.1
input_row = Input(shape=(num_of_words,))
#encoder = Dense(int(num_of_words/2), activation='relu')(input_row)
encoder= Dense(num_of_topics, activation='relu', activity_regularizer=keras.regularizers.l1_l2(regularization))(input_row)
#decoder = Dense(int(num_of_words/2), activation='relu')(encoder)
output_row = Dense(num_of_words,activation='sigmoid')(encoder)

autoencoder = Model(input_row,output_row)
#autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])#optimizer='adadelta', loss='mse', metrics=['accuracy'])
#autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history = autoencoder.fit(matrix,matrix,batch_size=256,epochs=100,validation_split=0.1)
weight_in = autoencoder.get_weights()[0]
weight_out = autoencoder.get_weights()[2]
blob = np.array([])
weight_in = weight_in.transpose()
num_of_important_words = 20
from results_saver import LogWriter

def get_extremes(matrix, num_of_topics, num_of_important_words,reverse_word_map,is_max,name,log_writer):

    topic_words = [[] for i in range(num_of_topics)]
    for i in range(num_of_topics):
        if is_max:
            extreme = (-np.sort(-matrix[i]))[num_of_important_words]
            indexes = np.argwhere(matrix[i]>=extreme)
        else:
            extreme = (np.sort(matrix[i]))[num_of_important_words]
            indexes = np.argwhere(matrix[i]<=extreme)
        for index in indexes:
            index = index[0]
            if index == 0:
                continue
            topic_words[i].append([reverse_word_map[index],matrix[i,index]])
    if is_max:
        topic_words = [sorted(words, key=lambda x: x[1], reverse=True) for words in topic_words]
    else:
        topic_words = [sorted(words, key=lambda x: x[1]) for words in topic_words]
    log_writer.write_2D_list(name, topic_words)

    #return topic_words


"""topic_words_in = [sorted(topic_words,key=lambda x: x[1],reverse=True) for topic_words in topic_words_in]
topic_words_out = [sorted(topic_words,key=lambda x: x[1],reverse=True) for topic_words in topic_words_out]
log_writer = LogWriter(log_file_desc='LDATestsRegularize{}'.format(regularization))
log_writer.write_2D_list('topic_words_in', topic_words_in)
log_writer.write_2D_list('topic_words_out', topic_words_out)"""

log_writer = LogWriter(log_file_desc='LDATestsRegularize{}'.format(regularization))
get_extremes(weight_in,num_of_topics,num_of_important_words,reverse_word_map,True,'topic_words_in_max',log_writer)
get_extremes(weight_in,num_of_topics,num_of_important_words,reverse_word_map,False,'topic_words_in_min',log_writer)
get_extremes(weight_out,num_of_topics,num_of_important_words,reverse_word_map,True,'topic_words_out_max',log_writer)
get_extremes(weight_out,num_of_topics,num_of_important_words,reverse_word_map,False,'topic_words_out_min',log_writer)

"""texts = [text.split() for text in documents]
dictionary = corpora.Dictionary(texts)
# TODO maybe test work with dict self.dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]"""
from lda_impl import Lda
model = Lda(num_of_topics,num_of_important_words,
            passes=25,
            iterations=25)
model.train(documents)
"""gensim.models.LdaModel(
doc_term_matrix,
num_topics=num_of_topics,
id2word=dictionary,
passes=2,
iterations=2)"""

print(extract_important_words(model.get_topics(),True))
log_writer.write_2D_list('topic_words_lda', extract_important_words(model.get_topics(), True).values(), 'w+')
from NeuralTopicMatrix import NeuralTopicMatrix
neural_lda_in = NeuralTopicMatrix(weight_in,reverse_word_map,num_of_topics,tokenizer)
neural_lda_out = NeuralTopicMatrix(weight_out,reverse_word_map,num_of_topics,tokenizer)
test_model(documents, labels, neural_lda_in, log_writer,'neural_lda_in')
test_model(documents, labels, neural_lda_out, log_writer,'neural_lda_out')
test_model(documents, labels, model, log_writer,'standard_lda')

#print(topic_words)



