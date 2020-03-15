import tensorflow as tf

#Keras to tensorflow20 aliases

#Keras itself
keras = tf.keras
#layers aliases
Dense = tf.keras.layers.Dense
Bidirectional = tf.keras.layers.Bidirectional
LSTM = tf.keras.layers.LSTM
Flatten = tf.keras.layers.Flatten
Input = tf.keras.layers.Input
Conv1D = tf.keras.layers.Conv1D
Embedding = tf.keras.layers.Embedding
RepeatVector = tf.keras.layers.RepeatVector
MaxPooling1D = tf.keras.layers.MaxPooling1D
SimpleRNN = tf.keras.layers.SimpleRNN
GRU = tf.keras.layers.GRU
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization

#utils aliases
to_categorical = tf.keras.utils.to_categorical
plot_model = tf.keras.utils.plot_model
Sequence = tf.keras.utils.Sequence

#models aliases
Model = tf.keras.models.Model
Sequential = tf.keras.models.Sequential

#preprocesing aliases
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences


#callbacks aliases
EarlyStopping = keras.callbacks.EarlyStopping