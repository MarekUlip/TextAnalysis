import operator

from sklearn.manifold import TSNE

from dataset_helper import Dataset_Helper, stp_wrds
from results_saver import LogWriter
from gensim import corpora
import matplotlib.pyplot as plt
from aliaser import *
import os
import sys
import numpy as np
import gensim
from collections import Counter
from gensim.models import CoherenceModel
from lda_impl import Lda
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from NeuralTopicMatrix import NeuralTopicMatrix
import tkinter as tk
from tkinter import simpledialog

def extract_important_words(topics, keep_values=True):
    d = {}
    i = 0
    for x in topics:
        a = x[1].replace(" ", "")
        a = a.replace("\"", "")
        d[i] = []
        for y in a.split("+"):
            if keep_values:
                tmp = y.split("*")
                tmp[0] = float(tmp[0])
                d[i].append(tmp)
            else:
                d[i].append(y.split("*")[1])
        i += 1
    return list(d.values())

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
        Connects model topic indexes to topic ids from dataset. Every model index is assigned and all topic ids are used. Note that this method should be used
        on balanced datasets because it would prefer smaller topic groups due to higher precentage confidence.
        :param model: model containing topics to connect
        """
    # t = model.get_topics()
    topic_indexes = {}
    topics_of_index = {}
    confidence = []
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
        for tp_num, val in connection_results.items():
            confidence.append([key,tp_num,val/len(value)])
    confidence = sorted(confidence, key=operator.itemgetter(2),reverse=True)
    associated_indexes = []
    associated_topics = []
    for conf in confidence:
        if conf[1] in associated_indexes or conf[0] in associated_topics:
            continue
        associated_indexes.append(conf[1])
        associated_topics.append(conf[0])
        log_writer.add_log('Connecting topic {} to model index {} based on highest unused confidence of {}'.format(conf[0],conf[1],conf[2]))
        topic_indexes[conf[0]] = conf[1]

    for key, value in topic_indexes.items():
        topics_of_index[value] = [key]
    print(topic_indexes)
    print(topics_of_index)
    return topic_indexes, topics_of_index

def connect_topic_id_to_topics_old(model, representants, log_writer):
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

def extract_words(word_probs, threshold = -100000.0):
    return [[word_prob[1] for word_prob in topic_group if word_prob[0]>threshold] for topic_group in word_probs]

def get_extremes(matrix, num_of_topics, num_of_important_words,reverse_word_map,is_max,name,log_writer,dataset_name):

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
            topic_words[i].append([matrix[i,index],reverse_word_map[index]])
    if is_max:
        topic_words = [sorted(words, key=lambda x: x[1], reverse=True) for words in topic_words]
    else:
        topic_words = [sorted(words, key=lambda x: x[1]) for words in topic_words]
    log_writer.write_2D_list('{}\\{}'.format(dataset_name,name), topic_words)


    return topic_words

def swap_weights_and_words(topic_words):
    return [[[topic[1],topic[0]] for topic in t]for t in topic_words]

def measureCoherence(topic_words, log_writer, dictionary, docs,name,dataset_name):
    create_word_cloud(topic_words, name, log_writer,dataset_name)
    coh_corpus = [text.split() for text in docs]
    topic_words = extract_words(topic_words)
    print(topic_words)
    coherence = CoherenceModel(topics=topic_words,texts=coh_corpus,dictionary=dictionary,coherence='c_v',processes=1)
    u_mass = CoherenceModel(topics=topic_words,texts=coh_corpus,dictionary=dictionary,coherence='u_mass',processes=1)
    log_writer.add_log('Coherence for model {} is {}. Its UMass is {}'.format(name,coherence.get_coherence(),u_mass.get_coherence()))
    coherences = coherence.get_coherence_per_topic()
    u_masses = coherence.get_coherence_per_topic()
    for i in range(len(topic_words)):
        log_writer.add_log('Coherence for model {} and topic number {} is {}. Its UMass is {}'.format(name, i,coherences[i],
                                                                                  u_masses[i]))

def create_word_cloud(topics, name, log_writer,dataset_name):
    topics = swap_weights_and_words(topics)
    if len(topics) <= 10:
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    else:
        cols = [color for name, color in mcolors.XKCD_COLORS.items()]

    cloud = WordCloud(stopwords=stp_wrds,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    dimension = 2
    while dimension**2 < len(topics):
        dimension+=1
    fig, axes = plt.subplots(dimension, dimension, figsize=(10, 10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        if i >= len(topics):
            break
        fig.add_subplot(ax)
        topic_words = dict(topics[i])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig(log_writer.get_plot_path(dataset_name,name))
    plt.clf()

def plot_clustering_chart(model, is_lda, docs, log_writer:LogWriter,chart_name,dataset_name, n_topics=4):
    if len(docs) > 12000:
        print('Creating clustering chart would take too much time. Returning')
        return
    topic_weights = []
    for i, doc in enumerate(docs):
        if is_lda:
            topic_weights.append([w for i,w in model.analyse_text(doc)])
        else:
            topic_weights.append([w for i,w in model.analyse_text(doc,True)])
    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    #arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca',n_iter=251)
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_file("{}\\{}\\{}-lines.html".format(log_writer.path,dataset_name,chart_name))
    if n_topics <= 10:
        mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    else:
        mycolors = np.array([color for name, color in mcolors.XKCD_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    show(plot)