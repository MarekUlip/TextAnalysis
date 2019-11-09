from aliaser import *
import numpy as np
max_features = 10000
maxlen = 20

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(
num_words=max_features)



x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

np.load = np_load_old

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train, y_train,
epochs=10,
batch_size=32,
validation_split=0.2)
print(model.evaluate(x_test,y_test))