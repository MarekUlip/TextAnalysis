from neural_networks.aliaser import *
from text_generators.training_text_generator_RNN import TrainingTextGeneratorRNN
from results_saver import *
import os
import sys
import tkinter as tk
from tkinter import simpledialog

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
root = tk.Tk()
root.withdraw()

preprocess = False
datasets_helper = Dataset_Helper(preprocess)
log_writer = LogWriter(log_file_desc=simpledialog.askstring(title="Test Name",
                                                            prompt="Insert test name:", initialvalue='LSTM_'))
results = []
num_of_words = 15000
datasets_helper.set_wanted_datasets([2])#[0,1,2,3,6,9])
while datasets_helper.next_dataset():
    log_writer.add_log("Starting testing dataset {}".format(datasets_helper.get_dataset_name()))
    tokenizer = Tokenizer(num_words=num_of_words)
    generator = datasets_helper.text_generator()
    log_writer.add_log("Starting preprocessing and tokenization.")
    tokenizer.fit_on_texts(generator)
    log_writer.add_log("Done. Building model now.")

    batch_size = 256
    gauss_noise = 0.7
    epochs = 40
    val_split = 0.2
    val_data_count = int(datasets_helper.get_num_of_train_texts()*val_split)
    tokenizer_mode = 'tfidf'
    
    model:Sequential = Sequential()
    enhanced_num_of_topics = 128#int(np.ceil(datasets_helper.get_num_of_topics()*2))#-datasets_helper.get_num_of_topics()/2))
    model.add(LSTM(128,input_shape=(1,num_of_words), return_sequences=True))
    #model.add(RepeatVector(3))
    model.add(keras.layers.GaussianNoise(gauss_noise))
    #model.add(Dropout(0.1))
    model.add(LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.LayerNormalization())
    model.add(Dropout(0.3))
    #model.add(keras.layers.GaussianNoise(0.2))
    model.add(LSTM(128))
    model.add(keras.layers.LayerNormalization())
    #model.add(Dense(enhanced_num_of_topics, activation='relu', input_shape=(num_of_words,)))
    #model.add(Dense(enhanced_num_of_topics, activation='relu'))
    model.add(Dense(datasets_helper.get_num_of_topics(),activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    log_writer.add_log("Done. Now lets get training.")
    log_writer.add_log(
        'Arguments used were: batch_size={}\nNum_of_epochs={}'.format(batch_size,  epochs))
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3,restore_best_weights=True)
    train = TrainingTextGeneratorRNN(
        datasets_helper.get_train_file_path(), batch_size,
        datasets_helper.get_num_of_train_texts()-val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=None, preprocess=preprocess, preload_dataset=True, is_predicting=False, tokenizer_mode=tokenizer_mode)
    validation = TrainingTextGeneratorRNN(
        datasets_helper.get_train_file_path(), batch_size,
        val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=None, preprocess=preprocess, preload_dataset=True, is_predicting=False, start_point=datasets_helper.get_num_of_train_texts()-val_data_count,tokenizer_mode=tokenizer_mode)
    history = model.fit(
        x=train,
        epochs=epochs,
        callbacks=[early_stop],
        validation_data=validation)
    test = TrainingTextGeneratorRNN(
        datasets_helper.get_test_file_path(), batch_size,
        datasets_helper.get_num_of_test_texts(),
        num_of_words, tokenizer, ";",
        datasets_helper, preprocess=preprocess, preload_dataset=True, is_predicting=False,tokenizer_mode=tokenizer_mode)
    result = model.evaluate(x=test)
    print(result)
    result.append(datasets_helper.get_dataset_name())
    #model.summary(print_fn=result.append)
    results.append(result)
    log_writer.add_log("Done. Finishing this dataset.")
    gnr = TrainingTextGeneratorRNN(
        datasets_helper.get_test_file_path(), batch_size,
        datasets_helper.get_num_of_test_texts(),
        num_of_words, tokenizer, ";",
        datasets_helper, preprocess=preprocess, preload_dataset=True, is_predicting=True, tokenizer_mode=tokenizer_mode)

    finish_dataset(model,gnr,datasets_helper,log_writer,history)
    log_writer.add_log("Finished testing dataset {}".format(datasets_helper.get_dataset_name()),True)

    log_writer.write_2D_list("results", results,'a+')
log_writer.end_logging()