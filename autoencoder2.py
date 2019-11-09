from helper_functions import Dataset_Helper
from results_saver import LogWriter
from gensim import corpora
import matplotlib.pyplot as plt
from aliaser import *
import os
import sys





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
dataset_helper.next_dataset()
documents = dataset_helper.get_texts_as_list()
tokenizer = Tokenizer(num_words=num_of_words)
tokenizer.fit_on_texts(documents)
matrix = tokenizer.texts_to_matrix(documents, mode='binary')
#mydict = corpora.Dictionary([line.split() for line in documents],prune_at=num_of_words)
#corpus = [mydict.doc2bow(line.split()) for line in documents]

#tfidf = TfidfModel(corpus)
#print(tfidf)

"""model = Sequential()
model.add(Dense(num_of_words*num_of_topics,activation='relu', input_shape=(num_of_words,)))
model.add(Dense(num_of_words,activation='sigmoid'))"""
input_row = Input(shape=(num_of_words,))
#encoder = Dense(int(num_of_words/2), activation='relu')(input_row)
encoder= Dense(int(num_of_words/num_of_topics), activation='relu')(input_row)
#decoder = Dense(int(num_of_words/2), activation='relu')(encoder)
output_row = Dense(num_of_words,activation='sigmoid')(encoder)

autoencoder = Model(input_row,output_row)
#autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])#optimizer='adadelta', loss='mse', metrics=['accuracy'])
#autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history = autoencoder.fit(matrix,matrix,batch_size=64,epochs=20,validation_split=0.1)

#topic_matrix = model.layers[0].weights
documents = dataset_helper.get_texts_as_list(dataset_helper.open_file_stream(dataset_helper.get_test_file_path()))

matrix = tokenizer.texts_to_matrix(documents)#vectorizer.transform(documents).todense()/num_of_words
result = autoencoder.evaluate(matrix,matrix)
print(result)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


"""plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()"""
#plot_model(autoencoder,results_saver.get_plot_path("","model-graph"),show_shapes=True)