import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from training_text_generator_RNN import Training_Text_Generator_RNN
from helper_functions import Dataset_Helper
from results_saver import LogWriter
import os
import sys
from aliaser import *

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

"""config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)"""

datasets_helper = Dataset_Helper()
results_saver = LogWriter(log_file_desc="GRU-Conv")
results = []
num_of_words = 10000

while datasets_helper.next_dataset():
    results_saver.add_log("Starting testing dataset {}".format(datasets_helper.get_dataset_name()))
    validation_count = datasets_helper.get_num_of_train_texts() // 10
    tokenizer = Tokenizer(num_words=num_of_words,
                         filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                         lower=False, split=' ')
    generator = datasets_helper.text_generator()
    results_saver.add_log("Starting preprocessing and tokenization.")
    tokenizer.fit_on_texts(generator)
    results_saver.add_log("Done. Building model now.")

    enhanced_num_of_topics = int(np.ceil(datasets_helper.get_num_of_topics()*2-datasets_helper.get_num_of_topics()/2))
    model = Sequential()
    model.add(Conv1D(40, 1, activation='relu',
                            input_shape=(1,num_of_words)))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(40, 1, activation='relu'))
    model.add(GRU(40, dropout=0.1, recurrent_dropout=0.5))
    model.add(Dense(datasets_helper.get_num_of_topics(),activation='softmax'))
    #model.add(Dense(enhanced_num_of_topics, activation='relu', input_shape=(num_of_words,)))
    #model.add(Dense(enhanced_num_of_topics, activation='relu'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    results_saver.add_log("Done. Now lets get training.")
    batch_size = 512
    history = model.fit_generator(generator=Training_Text_Generator_RNN(datasets_helper.get_train_file_path(), batch_size, datasets_helper.get_num_of_train_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics()), epochs=10, validation_data=Training_Text_Generator_RNN(datasets_helper.get_train_file_path(), batch_size, validation_count, num_of_words, tokenizer, ";", datasets_helper.get_num_of_topics(),start_point=datasets_helper.get_num_of_train_texts()-validation_count))
    #history = model.fit(x_train,y_train, epochs=8,batch_size=256,validation_data=(x_validation,y_valitadio))
    result = model.evaluate_generator(generator=Training_Text_Generator_RNN(datasets_helper.get_test_file_path(), batch_size, datasets_helper.get_num_of_test_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics()))# model.evaluate(test_sequences,test_labels)
    print(result)
    result.append(datasets_helper.get_dataset_name())
    model.summary(print_fn=result.append)
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

    results_saver.add_log("Finished testing dataset {}".format(datasets_helper.get_dataset_name()))

results_saver.write_2D_list("results",results)
results_saver.end_logging()
