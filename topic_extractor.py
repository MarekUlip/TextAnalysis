import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Bidirectional, LSTM, Embedding, Flatten
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from embedding_loader import get_embedding_matrix
from training_text_generator_RNN_embedding import Training_Text_Generator_RNN_Embedding
from keras.utils import plot_model
from helper_functions import Dataset_Helper
from results_saver import LogWriter
import os
import sys
from gensim.models import LdaModel
from text_generators.gensim_text_generator import GensimTextGenerator
from scipy import spatial


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

"""config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)"""
preprocess = True
datasets_helper = Dataset_Helper(preprocess=preprocess)

num_of_words = 10000
#embedding_dim = 50
max_len = 10

results_saver = LogWriter(log_file_desc="Topic-Extraction")
results = []


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
        d[i] = " ".join(d[i])
        i += 1
    return d

def get_topics(model, topic_word_count):
    """
    Get model topics with their words base on topic_word_count parameter.
    :return: model topics with their words.
    """
    return model.print_topics(-1, topic_word_count)

embedding_dim = 50

while datasets_helper.next_dataset():
    topic_names= ['world school people politics','sport victory match competition','business stock money price', 'software computers science engineering']
    results_saver.add_log("Starting testing dataset {}".format(datasets_helper.get_dataset_name()))
    gensim_text_generator = GensimTextGenerator(datasets_helper.get_train_file_path(),preprocess=preprocess)
    gensim_text_generator.init_dictionary()
    lda = LdaModel(corpus=gensim_text_generator.get_corpus(), id2word=gensim_text_generator.get_dictionary(), num_topics=datasets_helper.get_num_of_topics(),num_terms=num_of_words,passes=20,iterations=20)
    topic_words = extract_important_words(get_topics(lda,10),False)
    print(topic_words)
    validation_count = 500#datasets_helper.get_num_of_train_texts() // 10
    tokenizer = Tokenizer(num_words=num_of_words,
                         filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                         lower=False, split=' ')
    generator = datasets_helper.text_generator()
    results_saver.add_log("Starting preprocessing and tokenization.")
    tokenizer.fit_on_texts(generator)
    results_saver.add_log("Done. Building model now.")

    model = Sequential()
    enhanced_num_of_topics = int(np.ceil(datasets_helper.get_num_of_topics())*2.5) #-datasets_helper.get_num_of_topics()/2))
    model.add(Embedding(num_of_words, embedding_dim))
    #model.add(Flatten())
    model.layers[0].set_weights([get_embedding_matrix(num_of_words, embedding_dim, tokenizer.word_index)])
    model.layers[0].trainable = False
    results_saver.add_log("Compiling model")

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model,results_saver.get_plot_path("","model-graph"),show_shapes=True)
    results_saver.add_log("Done. Now lets get training.")
    datasets_helper.reset_file_stream()
    texts = tokenizer.texts_to_sequences(datasets_helper.get_texts_as_list())
    texts = pad_sequences(texts,max_len)
    batch_size = 128
    a = np.reshape(model.predict(texts[0]),-1)
    topic_embeddings =[]
    guessed_topic_embeddings = []
    for value in topic_words.values():
        text = pad_sequences(tokenizer.texts_to_sequences(value), max_len)
        guessed_topic_embeddings.append(np.reshape(model.predict(text),-1))

    for value in topic_names:
        text = pad_sequences(tokenizer.texts_to_sequences(value), max_len)
        topic_embeddings.append(np.reshape(model.predict(text),-1))

    for t_e in topic_embeddings:
        distances = []
        for g_t_e in guessed_topic_embeddings:
            distances.append(spatial.distance.cosine(t_e, g_t_e))
        print(distances.index(min(distances)))
    print(print(topic_words))
    #print(a)
    #callbacks = [keras.callbacks.TensorBoard(log_dir=datasets_helper.get_tensor_board_path())]
    #history = model.fit_generator( generator=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), batch_size, datasets_helper.get_num_of_train_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics(),max_len), epochs=10, validation_data=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), batch_size, validation_count, num_of_words, tokenizer, ";", datasets_helper.get_num_of_topics(),max_len,start_point=datasets_helper.get_num_of_train_texts()-validation_count))
    #history = model.fit(x_train,y_train, epochs=8,batch_size=256,validation_data=(x_validation,y_valitadio))
    #result = model.evaluate_generator(generator=Training_Text_Generator_RNN_Embedding(datasets_helper.get_test_file_path(), batch_size, datasets_helper.get_num_of_test_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics(),max_len))# model.evaluate(test_sequences,test_labels)
    """print(result)
    result.append(datasets_helper.get_dataset_name())
    #model.summary(print_fn=result.append)
    results.append(result)
    results_saver.add_log("Done. Finishing this dataset.")
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss {}'.format(datasets_helper.get_dataset_name()))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_saver.get_plot_path(datasets_helper.get_dataset_name(),"loss"))


    plt.clf()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy {}'.format(datasets_helper.get_dataset_name()))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_saver.get_plot_path(datasets_helper.get_dataset_name(),"acc"))
    plt.clf()

    results_saver.add_log("Finished testing dataset {}".format(datasets_helper.get_dataset_name()))"""

    results_saver.write_2D_list("results",results)
results_saver.end_logging()