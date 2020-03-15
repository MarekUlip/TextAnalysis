from neural_networks.aliaser import *
from text_generators.training_text_generator_RNN_embedding import TrainingTextGeneratorRNNEmbedding
from dataset_loader.dataset_helper import Dataset_Helper
from results_saver import LogWriter, finish_dataset
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
                                                            prompt="Insert test name:", initialvalue='LSTM_Embedding_Learned_'))
results = []
num_of_words = 7500
max_seq_len = 250
datasets_helper.set_wanted_datasets([3])
while datasets_helper.next_dataset():
    results_saver.add_log("Starting testing dataset {}".format(datasets_helper.get_dataset_name()))
    tokenizer = Tokenizer(num_words=num_of_words)
                         #filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                         #lower=False, split=' ')
    generator = datasets_helper.text_generator()
    results_saver.add_log("Starting preprocessing and tokenization.")
    tokenizer.fit_on_texts(generator)
    results_saver.add_log("Done. Building model now.")

    batch_size = 128
    epochs = 1
    val_split = 0.2
    val_data_count = int(datasets_helper.get_num_of_train_texts()*val_split)
    model:Sequential = Sequential()
    enhanced_num_of_topics = 128#int(np.ceil(datasets_helper.get_num_of_topics()*2))#-datasets_helper.get_num_of_topics()/2))
    model.add(Embedding(num_of_words,50,input_length=max_seq_len))
    #model.add(keras.layers.SpatialDropout1D(0.2))
    #model.add(Dropout(0.01))
    #model.add(keras.layers.GaussianNoise(0.3))
    #model.add(RepeatVector(3))
    #model.add(LSTM(enhanced_num_of_topics, return_sequences=True,))#activity_regularizer=keras.regularizers.l1(0.1)))
    #model.add(keras.layers.GaussianNoise(0.2))
    model.add(LSTM(enhanced_num_of_topics))#,activity_regularizer=keras.regularizers.l1(0.01)))
    #model.add(Dense(enhanced_num_of_topics, activation='relu', input_shape=(num_of_words,)))
    #model.add(Dense(enhanced_num_of_topics, activation='relu'))
    model.add(Dense(datasets_helper.get_num_of_topics(),activation='softmax'))
    #opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    results_saver.add_log("Done. Now lets get training.")
    results_saver.add_log(
        'Arguments used were: batch_size={}\nNum_of_epochs={}'.format(batch_size,  epochs))
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3)
    train = TrainingTextGeneratorRNNEmbedding(
        datasets_helper.get_train_file_path(), batch_size,
        datasets_helper.get_num_of_train_texts() - val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=max_seq_len, preprocess=preprocess, preload_dataset=True, is_predicting=False)
    validation = TrainingTextGeneratorRNNEmbedding(
        datasets_helper.get_train_file_path(), batch_size,
        val_data_count,
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=max_seq_len, preprocess=preprocess, preload_dataset=True, is_predicting=False,
        start_point=datasets_helper.get_num_of_train_texts() - val_data_count)
    history = model.fit(x=train,
                        epochs=epochs,
                        callbacks=[early_stop],
                        validation_data=validation)
    test = TrainingTextGeneratorRNNEmbedding(
        datasets_helper.get_test_file_path(), batch_size,
        datasets_helper.get_num_of_test_texts(),
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=max_seq_len, preprocess=preprocess, preload_dataset=True, is_predicting=False)
    result = model.evaluate(x=test)
    print(result)
    result.append(datasets_helper.get_dataset_name())
    results.append(result)
    results_saver.add_log("Done. Finishing this dataset.")
    gnr = TrainingTextGeneratorRNNEmbedding(
        datasets_helper.get_test_file_path(), batch_size,
        datasets_helper.get_num_of_test_texts(),
        num_of_words, tokenizer, ";",
        datasets_helper, max_len=max_seq_len, preprocess=preprocess, preload_dataset=True, is_predicting=True)
    finish_dataset(model,gnr,datasets_helper,results_saver,history)
    results_saver.add_log("Finished testing dataset {}".format(datasets_helper.get_dataset_name()),True)

    results_saver.write_2D_list("results",results,'a+')
results_saver.end_logging()