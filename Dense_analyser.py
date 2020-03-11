import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from training_text_generator import Training_Text_Generator
from dataset_helper import Dataset_Helper
from results_saver import LogWriter
import os
import sys
from aliaser import *

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

datasets_helper = Dataset_Helper(True)
datasets_helper.set_wanted_datasets([3])
results_saver = LogWriter(log_file_desc="Dense-with-preprocessing-tfidf")
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

    model = Sequential()
    enhanced_num_of_topics = 128#int(np.ceil(datasets_helper.get_num_of_topics()*2-datasets_helper.get_num_of_topics()/2))
    #model.add(Dense(40, activation='relu', input_shape=(num_of_words,)))
    #model.add(Dense(40, activation='relu'))
    model.add(Dense(enhanced_num_of_topics, activation='relu', input_shape=(num_of_words,)))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(Dense(enhanced_num_of_topics, activation='relu'))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.GaussianNoise(0.9))
    model.add(Dense(datasets_helper.get_num_of_topics(),activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    results_saver.add_log("Done. Now lets get training.")
    batch_size = 256
    history = model.fit(verbose=2,x=Training_Text_Generator(datasets_helper.get_train_file_path(), batch_size, datasets_helper.get_num_of_train_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics()),
                                  epochs=6, validation_data=Training_Text_Generator(datasets_helper.get_train_file_path(), batch_size, validation_count, num_of_words, tokenizer, ";", datasets_helper.get_num_of_topics(),start_point=datasets_helper.get_num_of_train_texts()-validation_count))
    #history = model.fit(x_train,y_train, epochs=8,batch_size=256,validation_data=(x_validation,y_valitadio))
    result = model.evaluate(x=Training_Text_Generator(datasets_helper.get_test_file_path(), batch_size, datasets_helper.get_num_of_test_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics()))# model.evaluate(test_sequences,test_labels)
    print(result)
    result.append(datasets_helper.get_dataset_name())
    model.summary(print_fn=result.append)
    #result.append(model.summary())
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

    results_saver.add_log("Finished testing dataset {}".format(datasets_helper.get_dataset_name()))

results_saver.write_2D_list("results",results)
results_saver.end_logging()