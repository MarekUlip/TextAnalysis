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

base_path = os.getcwd()
csv_folder = base_path + "\\datasets\\"
f = open(csv_folder+"7\\train.csv")
def text_generator():
    for text in f:
        if text == "":
            break
        s = text.split(";")
        if len(s) <= 1:
            break
        yield s[1]
    """with open(csv_folder+"7\\test.csv") as train_file:
        for text in train_file:
            yield text.split(";")[1]"""

config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)


num_of_words = 10000
validation_count = num_of_words // 10
train_set = deep_preprocessor.load_csv([csv_folder+"7\\train.csv"],";",shuffle=True)
test_set = deep_preprocessor.load_csv([csv_folder+"7\\test.csv"],";",shuffle=True)
tokenizer = Tokenizer(num_words=num_of_words,
                     filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                     lower=False, split=' ')
"""tokenizer.fit_on_texts(train_set[1])
tokenizer.fit_on_texts(test_set[1])
train_set = None
test_set = None"""
t = text_generator()
tokenizer.fit_on_texts(t)#)Simple_Text_Generator(csv_folder+"7\\train.csv",4096,5000,';'))
#tokenizer.fit_on_texts(Simple_Text_Generator(csv_folder+"7\\test.csv",4096,5000,';'))

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(num_of_words,)))
model.add(Dense(12, activation='relu'))
model.add(Dense(8,activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
batch_size = 512
history = model.fit_generator(generator=Training_Text_Generator(csv_folder+"7\\train.csv",batch_size,75000,num_of_words,tokenizer,";"),epochs=3,validation_data=Training_Text_Generator(csv_folder+"7\\train.csv",batch_size,5000,num_of_words,tokenizer,";",75000))
#history = model.fit(x_train,y_train, epochs=8,batch_size=256,validation_data=(x_validation,y_valitadio))
results = model.evaluate_generator(generator=Training_Text_Generator(csv_folder+"7\\test.csv",batch_size,10000,num_of_words,tokenizer,";"))# model.evaluate(test_sequences,test_labels)
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