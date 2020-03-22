from enum import Enum

from neural_networks.aliaser import *
from text_generators.training_text_generator_RNN import TrainingTextGeneratorRNN
from text_generators.training_text_generator import TrainingTextGenerator
from text_generators.training_text_generator_RNN_embedding import TrainingTextGeneratorRNNEmbedding
from results_saver import *
import os
import sys
import tkinter as tk
from tkinter import simpledialog
from neural_networks.embedding_loader import get_embedding_matrix

class ModelType(Enum):
    DENSE = 0
    LSTM = 1
    RNN = 2
    GRU = 3
    BIDI_GRU = 4
    CONV = 5
    CONV_GRU = 6
    EMBEDDING_GLOVE_LSTM = 7
    EMBEDDING_GLOVE_TRAINED_LSTM = 8
    EMBEDDING_TRAINED_LSTM = 9

def get_LSTM_model(datasets_helper, params=None):
    model: Sequential = Sequential()
    model.add(LSTM(128, input_shape=(1, num_of_words), return_sequences=True))
    model.add(keras.layers.GaussianNoise(0.6))
    model.add(LSTM(128, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.LayerNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(keras.layers.LayerNormalization())
    model.add(Dense(datasets_helper.get_num_of_topics(), activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_dense_model(datasets_helper, params=None):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(num_of_words,)))
    #model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.GaussianNoise(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(datasets_helper.get_num_of_topics(), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_RNN_model(datasets_helper, params=None):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(1, num_of_words), return_sequences=True))
    model.add(keras.layers.GaussianNoise(0.3))
    model.add(SimpleRNN(64))
    model.add(keras.layers.LayerNormalization())
    model.add(Dense(datasets_helper.get_num_of_topics(), activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_GRU_model(datasets_helper, params=None):
    model = Sequential()
    model.add(GRU(64, input_shape=(1, num_of_words), return_sequences=True))
    model.add(keras.layers.GaussianNoise(0.3))
    model.add(GRU(64))
    model.add(keras.layers.LayerNormalization())
    model.add(Dense(datasets_helper.get_num_of_topics(), activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_conv_model(datasets_helper, params=None):
    embedding_dim = params.get('max_seq_len',200)
    tokenizer = params.get('tokenizer',None)
    max_seq_len = params.get('max_seq_len',400)

    model = Sequential()
    model.add(Embedding(num_of_words, embedding_dim, input_length=max_seq_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(keras.layers.GaussianNoise(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(keras.layers.LayerNormalization())
    model.add(Dense(datasets_helper.get_num_of_topics(), activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_conv_gru_model(datasets_helper, params=None):
    model = Sequential()
    model.add(Conv1D(128, 1, activation='relu',
                            input_shape=(1,num_of_words)))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(128, 1, activation='relu'))
    model.add(keras.layers.GaussianNoise(0.4))
    model.add(GRU(128))
    model.add(keras.layers.LayerNormalization())
    model.add(Dense(datasets_helper.get_num_of_topics(),activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_bidi_model(datasets_helper, params=None):
    model = Sequential()
    enhanced_num_of_topics = 64  # int(np.ceil(datasets_helper.get_num_of_topics()*4)) #-datasets_helper.get_num_of_topics()/2))
    model.add(Bidirectional(LSTM(enhanced_num_of_topics, return_sequences=True), input_shape=(1, num_of_words)))
    model.add(keras.layers.GaussianNoise(0.3))
    #model.add(Bidirectional(LSTM(enhanced_num_of_topics, return_sequences=True)))
    model.add(Bidirectional(LSTM(enhanced_num_of_topics)))
    #model.add(keras.layers.LayerNormalization())
    model.add(Dense(datasets_helper.get_num_of_topics(), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_embedding_glove_model(datasets_helper, params=None,train_embedding=False):
    embedding_dim = params.get('embedding_dim',50)
    max_seq_len = params.get('max_seq_len')
    tokenizer = params.get('tokenizer',None)
    model = Sequential()
    model.add(Embedding(num_of_words, embedding_dim, input_length=max_seq_len))
    model.add(keras.layers.GaussianNoise(0.4))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(keras.layers.LayerNormalization())
    model.add(Dense(datasets_helper.get_num_of_topics(), activation='softmax'))
    model.layers[0].set_weights([get_embedding_matrix(num_of_words, embedding_dim, tokenizer.word_index)])
    model.layers[0].trainable = train_embedding

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def get_embedding_trained_model(datasets_helper, params=None):
    embedding_dim = params.get('embedding_dim',50)
    max_seq_len = params.get('max_seq_len')
    model: Sequential = Sequential()
    model.add(Embedding(num_of_words, embedding_dim, input_length=max_seq_len))
    model.add(keras.layers.GaussianNoise(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(keras.layers.LayerNormalization())
    model.add(Dense(datasets_helper.get_num_of_topics(), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_text_generator_class(model_type, params=None):
    if model_type in [ModelType.DENSE]:
        return TrainingTextGenerator
    if model_type in [ModelType.RNN,ModelType.GRU,ModelType.LSTM,ModelType.BIDI_GRU,  ModelType.CONV_GRU]:
        return TrainingTextGeneratorRNN
    if model_type in [ModelType.CONV, ModelType.EMBEDDING_GLOVE_LSTM, ModelType.EMBEDDING_GLOVE_TRAINED_LSTM, ModelType.EMBEDDING_TRAINED_LSTM]:
        return TrainingTextGeneratorRNNEmbedding
    return TrainingTextGenerator

def get_model_from_type(model_type, datasets_helper, params=None):
    if model_type == ModelType.LSTM:
        return get_LSTM_model(datasets_helper,params)
    elif model_type == ModelType.DENSE:
        return get_dense_model(datasets_helper,params)
    elif model_type == ModelType.BIDI_GRU:
        return get_bidi_model(datasets_helper,params)
    elif model_type == ModelType.CONV:
        return get_conv_model(datasets_helper,params)
    elif model_type == ModelType.CONV_GRU:
        return get_conv_gru_model(datasets_helper,params)
    elif model_type == ModelType.GRU:
        return get_GRU_model(datasets_helper,params)
    elif model_type == ModelType.RNN:
        return get_RNN_model(datasets_helper,params)
    elif model_type == ModelType.EMBEDDING_TRAINED_LSTM:
        return get_embedding_trained_model(datasets_helper,params)
    elif model_type == ModelType.EMBEDDING_GLOVE_LSTM:
        return get_embedding_glove_model(datasets_helper, params)
    elif model_type == ModelType.EMBEDDING_GLOVE_TRAINED_LSTM:
        return get_embedding_glove_model(datasets_helper, params,True)
    return Sequential()

tested_model = ModelType.LSTM
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
root = tk.Tk()
root.withdraw()

preprocess = True
datasets_helper = Dataset_Helper(preprocess)
log_writer = LogWriter(log_file_desc=simpledialog.askstring(title="Test Name",
                                                            prompt="Insert test name:", initialvalue='{}_{}'.format(tested_model.name,'prep_' if preprocess else 'no-prep_')),result_desc='debug')
results = []
num_of_words = 15000
batch_size = 256
epochs = 50
val_split = 0.2
max_seq_len = 400
embedding_dim = 200
tokenizer_mode = 'binary'


datasets_helper.set_wanted_datasets([9,14])
while datasets_helper.next_dataset():
    val_data_count = int(datasets_helper.get_num_of_train_texts() * val_split)
    log_writer.add_log("Starting testing dataset {}".format(datasets_helper.get_dataset_name()))
    tokenizer = Tokenizer(num_words=num_of_words)
    generator = datasets_helper.text_generator()
    log_writer.add_log("Starting preprocessing and tokenization.")
    tokenizer.fit_on_texts(generator)
    log_writer.add_log("Done. Building model now.")
    params = {}
    params['tokenizer'] = tokenizer
    params['max_seq_len'] = max_seq_len
    params['embedding_dim'] = embedding_dim
    model = get_model_from_type(tested_model,datasets_helper,params)
    model.summary()

    log_writer.add_log("Done. Preparing generators.")
    log_writer.add_log(
        'Arguments used were: batch_size={}\nNum_of_epochs={}\nembedding_dim={}\ntokenizer_mode={}'.format(batch_size, epochs,max_seq_len,tokenizer_mode))
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    text_generator = get_text_generator_class(tested_model)
    train = text_generator(
        datasets_helper.get_train_file_path(), batch_size,
        datasets_helper.get_num_of_train_texts() - val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=max_seq_len, preprocess=preprocess, preload_dataset=True, is_predicting=False,
        tokenizer_mode=tokenizer_mode)
    validation = text_generator(
        datasets_helper.get_train_file_path(), batch_size,
        val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=max_seq_len, preprocess=preprocess, preload_dataset=True, is_predicting=False,
        start_point=datasets_helper.get_num_of_train_texts() - val_data_count, tokenizer_mode=tokenizer_mode)
    log_writer.add_log('Done. Starting training.')
    history = model.fit(
        x=train,
        epochs=epochs,
        callbacks=[early_stop],
        validation_data=validation)
    test = text_generator(
        datasets_helper.get_test_file_path(), batch_size,
        datasets_helper.get_num_of_test_texts(),
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=max_seq_len, preprocess=preprocess, preload_dataset=True, is_predicting=False,
        tokenizer_mode=tokenizer_mode)
    log_writer.add_log('Training done. Starting testing.')
    result = model.evaluate(x=test)
    print(result)
    result.append(datasets_helper.get_dataset_name())
    # model.summary(print_fn=result.append)
    results.append(result)
    log_writer.add_log("Done. Finishing this dataset.")
    gnr = text_generator(
        datasets_helper.get_test_file_path(), batch_size,
        datasets_helper.get_num_of_test_texts(),
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=max_seq_len,preprocess=preprocess, preload_dataset=False, is_predicting=True, tokenizer_mode=tokenizer_mode)

    finish_dataset(model, gnr, datasets_helper, log_writer, history)
    log_writer.add_log("Finished testing dataset {}".format(datasets_helper.get_dataset_name()), True)

    log_writer.write_2D_list("results", results, 'a+')
    results.clear()
log_writer.end_logging()


