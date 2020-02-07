from aliaser import *
import matplotlib.pyplot as plt
from matplotlib import figure
from training_text_generator_RNN_embedding import Training_Text_Generator_RNN_Embedding
from helper_functions import Dataset_Helper
from results_saver import LogWriter
import os
import sys
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


datasets_helper = Dataset_Helper(True)
results_saver = LogWriter(log_file_desc="LSTM-embedding-128neurons-base-prep-shuffled")
results = []
num_of_words = 7500
max_seq_len = 250
datasets_helper.set_wanted_datasets([3])
while datasets_helper.next_dataset():
    results_saver.add_log("Starting testing dataset {}".format(datasets_helper.get_dataset_name()))
    validation_count = 500#datasets_helper.get_num_of_train_texts() // 10
    tokenizer = Tokenizer(num_words=num_of_words)
                         #filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                         #lower=False, split=' ')
    generator = datasets_helper.text_generator()
    results_saver.add_log("Starting preprocessing and tokenization.")
    tokenizer.fit_on_texts(generator)
    results_saver.add_log("Done. Building model now.")

    batch_size = 128
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
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model,results_saver.get_plot_path("","model-graph"),show_shapes=True)
    results_saver.add_log("Done. Now lets get training.")
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3)
    history = model.fit_generator(generator=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), batch_size, datasets_helper.get_num_of_train_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics(),max_len=max_seq_len),
                                  epochs=30,
                                  callbacks=[early_stop],
                                  validation_data=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), batch_size, validation_count, num_of_words, tokenizer, ";", datasets_helper.get_num_of_topics(),start_point=datasets_helper.get_num_of_train_texts()-validation_count,max_len=max_seq_len))
    #history = model.fit(x_train,y_train, epochs=8,batch_size=256,validation_data=(x_validation,y_valitadio))
    result = model.evaluate_generator(generator=Training_Text_Generator_RNN_Embedding(datasets_helper.get_test_file_path(), batch_size, datasets_helper.get_num_of_test_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics(),max_len=max_seq_len))# model.evaluate(test_sequences,test_labels)
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
    """gnr = Training_Text_Generator_RNN_Embedding(datasets_helper.get_test_file_path(), batch_size, datasets_helper.get_num_of_test_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics(),max_len=max_seq_len)

    predicts = model.predict_generator(generator=gnr)
    predicts = predicts.argmax(axis=-1)
    labels = gnr.labels[:len(predicts)]#datasets_helper.get_labels(datasets_helper.get_test_file_path())
    #print(confusion_matrix(labels[:len(predicts)],predicts))


    cm = confusion_matrix(labels, predicts)"""
    """cm_df = pd.DataFrame(cm)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_df, annot=True)
    plt.savefig(results_saver.get_plot_path(datasets_helper.get_dataset_name(), 'confusion_matrix'))"""
    #print(cm)
    """fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')"""
                #bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    """ plt.title('Confusion matrix of the classifier')
    #fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    #plt.show()
    plt.savefig(results_saver.get_plot_path(datasets_helper.get_dataset_name(),'confusion_matrix'))
    """
    """plt.clf()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy {}'.format(datasets_helper.get_dataset_name()))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_saver.get_plot_path(datasets_helper.get_dataset_name(),"acc"))
    plt.clf()"""

    results_saver.add_log("Finished testing dataset {}".format(datasets_helper.get_dataset_name()))

results_saver.write_2D_list("results",results)
results_saver.end_logging()