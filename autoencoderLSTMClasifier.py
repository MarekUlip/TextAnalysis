#from __future__ import absolute_import, division, print_function, unicode_literals


from dataset_helper import Dataset_Helper
from results_saver import LogWriter
from gensim import corpora
import matplotlib.pyplot as plt
from aliaser import *
import os
import sys
import numpy as np
from sklearn.preprocessing import normalize





file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
#sess = tf.compat.v1.Session(config=config)
#tf.keras.backend.set_session(sess)
#results_saver = LogWriter(log_file_desc="Autoencoder")
results = []

num_of_words = 10000
num_of_topics = 4
dataset_helper = Dataset_Helper(True)
dataset_helper.set_wanted_datasets([3])
dataset_helper.next_dataset()
documents = dataset_helper.get_texts_as_list()
tokenizer = Tokenizer(num_words=num_of_words)
tokenizer.fit_on_texts(documents)
matrix = normalize(tokenizer.texts_to_matrix(documents, mode='tfidf'),'l1')

#mydict = corpora.Dictionary([line.split() for line in documents],prune_at=num_of_words)
#corpus = [mydict.doc2bow(line.split()) for line in documents]

#tfidf = TfidfModel(corpus)
#print(tfidf)
encoding_dim = 128
"""model = Sequential()
model.add(Dense(num_of_words*num_of_topics,activation='relu', input_shape=(num_of_words,)))
model.add(Dense(num_of_words,activation='sigmoid'))"""
input_row = Input(shape=(num_of_words,))
#encoder = Dense(int(num_of_words/2), activation='relu')(input_row)
encoder= Dense(int(encoding_dim), activation='relu')(input_row)
#decoder = Dense(int(num_of_words/2), activation='relu')(encoder)
output_row = Dense(num_of_words,activation='softmax')(encoder)

autoencoder = Model(input_row,output_row)
#autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])#optimizer='adadelta', loss='mse', metrics=['accuracy'])
#autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history = autoencoder.fit(matrix,matrix,batch_size=64,epochs=100,validation_split=0.2)
print(history)
#topic_matrix = model.layers[0].weights
from aliaser import to_categorical

labels = to_categorical(dataset_helper.get_labels(dataset_helper.get_train_file_path()),dataset_helper.get_num_of_topics())
encoder = Model(input_row,encoder)


#matrix = matrix.reshape((matrix.shape[1],matrix.shape[0]))
#for i in range(matrix.shape[0]):
#    print((autoencoder.predict(matrix[:,i])))
#encoder.compile()
encoded = encoder.predict(matrix)
#decoder.compile(optimizer='adam', loss='mse')
#decoded = decoder.predict(encoded)

#datasets_helper = Dataset_Helper(True)
results_saver = LogWriter(log_file_desc="20news_autoenc_dense")
num_of_words = 10000

model = Sequential()
enhanced_num_of_topics = 128#int(np.ceil(datasets_helper.get_num_of_topics()*2-datasets_helper.get_num_of_topics()/2))
#model.add(Dense(40, activation='relu', input_shape=(num_of_words,)))
#model.add(Dense(40, activation='relu'))
model.add(Dense(enhanced_num_of_topics, activation='relu', input_shape=(encoding_dim,)))
#model.add(keras.layers.LayerNormalization())
#model.add(keras.layers.Dropout(0.5))
model.add(Dense(enhanced_num_of_topics, activation='relu'))
#model.add(keras.layers.LayerNormalization())
#model.add(keras.layers.GaussianNoise(0.9))
model.add(Dense(dataset_helper.get_num_of_topics(),activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
results_saver.add_log("Done. Now lets get training.")
batch_size = 32
history = model.fit(encoded,labels,batch_size,epochs=100,validation_split=0.2)
#history = model.fit(x_train,y_train, epochs=8,batch_size=256,validation_data=(x_validation,y_valitadio))
encoded = encoder.predict(tokenizer.texts_to_matrix(dataset_helper.get_texts_as_list(dataset_helper.open_file_stream(dataset_helper.get_test_file_path()) ),mode='binary'))
labels = to_categorical(dataset_helper.get_labels(dataset_helper.get_test_file_path()),dataset_helper.get_num_of_topics())
result = model.evaluate(encoded,labels,batch_size)
print(result)
result.append(dataset_helper.get_dataset_name())
model.summary(print_fn=result.append)
#result.append(model.summary())
results.append(result)
results_saver.add_log("Done. Finishing this dataset.")
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss {}'.format(dataset_helper.get_dataset_name()))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(results_saver.get_plot_path(dataset_helper.get_dataset_name(),"loss"))


plt.clf()

results_saver.add_log("Finished testing dataset {}".format(dataset_helper.get_dataset_name()))

results_saver.write_2D_list("results",results)
results_saver.end_logging()
