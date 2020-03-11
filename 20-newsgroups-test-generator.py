import numpy as np
import os
from aliaser import *
import sys

from dataset_helper import Dataset_Helper
from results_saver import LogWriter
from training_text_generator_RNN_embedding import Training_Text_Generator_RNN_Embedding

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 3000

results_saver = LogWriter(log_file_desc="20NewsGlove100TestWithGenerator")
results = []

#categories = ['alt.atheism', 'soc.religion.christian']

print("START")
# Loading the data set - training data.
from sklearn.datasets import fetch_20newsgroups

#newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True )
datasets_helper = Dataset_Helper(preprocess=False)
# You can check the target names (categories) and some data files by following commands.
#print(newsgroups_train.target_names)  # prints all the categories

#print("\n".join(newsgroups_train.data[0].split("\n")[:3]))  # prints first line of the first data file
#print(newsgroups_train.target_names)
#print(len(newsgroups_train.data))

#labels = newsgroups_train.target
#texts = newsgroups_train.data

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
generator = datasets_helper.text_generator()
datasets_helper.next_dataset()
tokenizer.fit_on_texts(generator)
#sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
"""
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]

y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]

y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set ')

print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True )
t_labels = newsgroups_test.target
t_texts = newsgroups_test.data
t_sequences = tokenizer.texts_to_sequences(t_texts)

t_data = pad_sequences(t_sequences, maxlen=MAX_SEQUENCE_LENGTH)

t_labels = to_categorical(np.asarray(t_labels))"""

base_path = os.getcwd()

GLOVE_DIR = base_path + "\\embeddings"

embeddings_index = {}

f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8")

for line in f:
    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)

l_pool1 = MaxPooling1D(5)(l_cov1)

l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)

l_pool2 = MaxPooling1D(5)(l_cov2)

l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)

l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling

l_flat = Flatten()(l_pool3)

l_dense = Dense(128, activation='relu')(l_flat)

preds = Dense(20, activation='softmax')(l_dense)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['acc'])

print("model fitting - simplified convolutional neural network")

model.summary()

#print(x_train)
#print(y_train)

model.fit(x=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), 128, datasets_helper.get_num_of_train_texts(), MAX_NB_WORDS, tokenizer, ";",datasets_helper.get_num_of_topics(),MAX_SEQUENCE_LENGTH), epochs=10, validation_data=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), 128, VALIDATION_SPLIT, MAX_NB_WORDS, tokenizer, ";", datasets_helper.get_num_of_topics(),MAX_SEQUENCE_LENGTH,start_point=datasets_helper.get_num_of_train_texts()-VALIDATION_SPLIT))

result = model.evaluate(x=Training_Text_Generator_RNN_Embedding(datasets_helper.get_test_file_path(), 128, datasets_helper.get_num_of_test_texts(), MAX_NB_WORDS, tokenizer, ";",datasets_helper.get_num_of_topics(),MAX_SEQUENCE_LENGTH))
print(result)
results.append(result)

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)

# applying a more complex convolutional approach

convs = []

filter_sizes = [3, 4, 5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)

    l_pool = MaxPooling1D(5)(l_conv)

    convs.append(l_pool)

l_merge = keras.layers.Concatenate(axis=1)(convs)

l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)

l_pool1 = MaxPooling1D(5)(l_cov1)

l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)

l_pool2 = MaxPooling1D(30)(l_cov2)

l_flat = Flatten()(l_pool2)

l_dense = Dense(128, activation='relu')(l_flat)

preds = Dense(20, activation='softmax')(l_dense)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['acc'])

print("model fitting - more complex convolutional neural network")

model.summary()

model.fit(x=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), 50, datasets_helper.get_num_of_train_texts(), MAX_NB_WORDS, tokenizer, ";",datasets_helper.get_num_of_topics(),MAX_SEQUENCE_LENGTH), epochs=10, validation_data=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), 50, VALIDATION_SPLIT, MAX_NB_WORDS, tokenizer, ";", datasets_helper.get_num_of_topics(),MAX_SEQUENCE_LENGTH,start_point=datasets_helper.get_num_of_train_texts()-VALIDATION_SPLIT))





result = model.evaluate(x=Training_Text_Generator_RNN_Embedding(datasets_helper.get_test_file_path(), 128, datasets_helper.get_num_of_test_texts(), MAX_NB_WORDS, tokenizer, ";",datasets_helper.get_num_of_topics(),MAX_SEQUENCE_LENGTH))

print(result)
results.append(result)
results_saver.write_2D_list("results",results)