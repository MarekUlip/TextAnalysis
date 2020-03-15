from text_generators.training_text_generator import TrainingTextGenerator
from dataset_loader.dataset_helper import Dataset_Helper
from results_saver import LogWriter, finish_dataset
import os
import sys
from neural_networks.aliaser import *
import tkinter as tk
from tkinter import simpledialog

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
root = tk.Tk()
root.withdraw()

preprocess = True
datasets_helper = Dataset_Helper(preprocess)
datasets_helper.set_wanted_datasets([3])
results_saver = LogWriter(log_file_desc=simpledialog.askstring(title="Test Name",
                                                            prompt="Insert test name:", initialvalue='Dense_'))
results = []
num_of_words = 10000

while datasets_helper.next_dataset():
    results_saver.add_log("Starting testing dataset {}".format(datasets_helper.get_dataset_name()))
    tokenizer = Tokenizer(num_words=num_of_words)
    generator = datasets_helper.text_generator()
    results_saver.add_log("Starting preprocessing and tokenization.")
    tokenizer.fit_on_texts(generator)
    results_saver.add_log("Done. Building model now.")


    epochs = 1
    batch_size = 256
    val_split = 0.2
    val_data_count = int(datasets_helper.get_num_of_train_texts()*val_split)
    enhanced_num_of_topics = 128
    gauss_noise = 0.5
    tokenizer_mode = 'tfidf'

    model = Sequential()

    model.add(Dense(enhanced_num_of_topics, activation='relu', input_shape=(num_of_words,)))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(Dense(enhanced_num_of_topics, activation='relu'))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.GaussianNoise(gauss_noise))
    model.add(Dense(datasets_helper.get_num_of_topics(),activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    results_saver.add_log("Done. Now lets get training.")
    results_saver.add_log(
        'Arguments used were: batch_size={}\nNum_of_epochs={}'.format(batch_size, epochs))
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    train = TrainingTextGenerator(
        datasets_helper.get_train_file_path(), batch_size,
        datasets_helper.get_num_of_train_texts() - val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=None, preprocess=preprocess, preload_dataset=True, is_predicting=False, tokenizer_mode=tokenizer_mode)
    validation = TrainingTextGenerator(
        datasets_helper.get_train_file_path(), batch_size,
        val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=None, preprocess=preprocess, preload_dataset=True, is_predicting=False,
        start_point=datasets_helper.get_num_of_train_texts() - val_data_count, tokenizer_mode=tokenizer_mode)
    history = model.fit(verbose=2,
                        x=train,
                        epochs=epochs,
                        validation_data=validation)
    test = TrainingTextGenerator(
        datasets_helper.get_test_file_path(), batch_size,
        datasets_helper.get_num_of_test_texts(),
        num_of_words, tokenizer, ";",
        datasets_helper, preprocess=preprocess, preload_dataset=True, is_predicting=False, tokenizer_mode=tokenizer_mode)
    result = model.evaluate(x=test)
    print(result)
    result.append(datasets_helper.get_dataset_name())
    #result.append(model.summary())
    results.append(result)
    results_saver.add_log("Done. Finishing this dataset.")
    gnr = TrainingTextGenerator(
        datasets_helper.get_test_file_path(), batch_size,
        datasets_helper.get_num_of_test_texts(),
        num_of_words, tokenizer, ";",
        datasets_helper, preprocess=preprocess, preload_dataset=True, is_predicting=True, tokenizer_mode=tokenizer_mode)

    finish_dataset(model, gnr, datasets_helper, results_saver, history)

    results_saver.add_log("Finished testing dataset {}".format(datasets_helper.get_dataset_name()),True)

    results_saver.write_2D_list("results",results,'a+')
results_saver.end_logging()