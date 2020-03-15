from neural_networks.aliaser import *
import numpy as np
import matplotlib.pyplot as plt
from text_generators.training_text_generator_RNN import TrainingTextGeneratorRNN
from dataset_loader.dataset_helper import Dataset_Helper
from results_saver import LogWriter
import os
import sys
import tkinter as tk
from tkinter import simpledialog

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
root = tk.Tk()
root.withdraw()

preprocess = True
datasets_helper = Dataset_Helper(preprocess)
results_saver = LogWriter(log_file_desc=simpledialog.askstring(title="Test Name",
                                                            prompt="Insert test name:", initialvalue='GRU_'))
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

    epochs = 1
    val_split = 0.2
    batch_size = 512
    val_data_count = int(datasets_helper.get_num_of_train_texts() * val_split)
    enhanced_num_of_topics = 128
    gauss_noise = 0.5
    tokenizer_mode = 'tfidf'

    model = Sequential()
    model.add(GRU(64,dropout=0.2,recurrent_dropout=0.5,input_shape=(1,num_of_words),return_sequences=True))
    model.add(GRU(64,activation='relu'))
    #model.add(Dense(enhanced_num_of_topics, activation='relu', input_shape=(num_of_words,)))
    #model.add(Dense(enhanced_num_of_topics, activation='relu'))
    model.add(Dense(datasets_helper.get_num_of_topics(),activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    results_saver.add_log("Done. Now lets get training.")
    results_saver.add_log(
        'Arguments used were: batch_size={}\nNum_of_epochs={}'.format(batch_size,  epochs))
    train = TrainingTextGeneratorRNN(
        datasets_helper.get_train_file_path(), batch_size,
        datasets_helper.get_num_of_train_texts() - val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=None, preprocess=preprocess, preload_dataset=True, is_predicting=False,
        tokenizer_mode=tokenizer_mode)
    validation = TrainingTextGeneratorRNN(
        datasets_helper.get_train_file_path(), batch_size,
        val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=None, preprocess=preprocess, preload_dataset=True, is_predicting=False,
        start_point=datasets_helper.get_num_of_train_texts() - val_data_count, tokenizer_mode=tokenizer_mode)

    early_stop = EarlyStopping(monitor='val_accuracy', patience=3,restore_best_weights=True)
    history = model.fit(
        x=train,
        epochs=epochs,
        callbacks=[early_stop],
        validation_data=validation)
    test = TrainingTextGeneratorRNN(
        datasets_helper.get_test_file_path(), batch_size,
        datasets_helper.get_num_of_test_texts(),
        num_of_words, tokenizer, ";",
        datasets_helper, preprocess=preprocess, preload_dataset=True, is_predicting=False,
        tokenizer_mode=tokenizer_mode)
    result = model.evaluate(x=test)
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