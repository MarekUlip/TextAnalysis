import deep_preprocessor
import keras
import tensorflow as tf
from keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
import os
from training_text_generator import Training_Text_Generator
from simple_text_generator import Simple_Text_Generator

def vectorize_sequences(sequences, dimension=20000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

base_path = os.getcwd()
csv_folder = base_path + "\\datasets\\"

train_set = deep_preprocessor.load_csv([csv_folder+"7\\train.csv"],";",shuffle=True)
test_set = deep_preprocessor.load_csv([csv_folder+"7\\test.csv"],";",shuffle=True)
num_of_words = 5000
validation_count = 200
tokenizer = Tokenizer(num_words=num_of_words,
                     filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                     lower=False, split=' ')
tokenizer.fit_on_texts(Simple_Text_Generator(csv_folder+"7\\train.csv",4096,5000,';'))
tokenizer.fit_on_texts(Simple_Text_Generator(csv_folder+"7\\test.csv",4096,5000,';'))
#train_sequences = tokenizer.texts_to_sequences(train_set[1])
train_sequences = vectorize_sequences(tokenizer.texts_to_sequences(train_set[1]), num_of_words).astype(np.uint8)
train_labels = to_categorical(train_set[0]).astype(np.uint8)
#test_sequences = tokenizer.texts_to_sequences(test_set[1])
test_sequences = vectorize_sequences(tokenizer.texts_to_sequences(test_set[1]), num_of_words).astype(np.uint8)
test_labels = to_categorical(test_set[0]).astype(np.uint8)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(num_of_words,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8,activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_validation = train_sequences[:validation_count]
x_train = train_sequences[validation_count:]
y_valitadio = train_labels[:validation_count]
y_train = train_labels[validation_count:]

history = model.fit(x_train,y_train, epochs=8,batch_size=128,validation_data=(x_validation,y_valitadio))
results = model.evaluate(test_sequences,test_labels)
print(results)

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


plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()