from neural_networks.aliaser import *
import matplotlib.pyplot as plt

from neural_networks.embedding_loader import get_embedding_matrix
from training_text_generator_RNN_embedding import Training_Text_Generator_RNN_Embedding
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


datasets_helper = Dataset_Helper(preprocess=False)

num_of_words = 10000
#embedding_dim = 50
max_len = 300
for embedding_dim in [100]:
    log_writer = LogWriter(log_file_desc=simpledialog.askstring(title="Test Name",
                                                            prompt="Insert test name:", initialvalue="RNN_GloVe{}_".format(embedding_dim)))
    results = []

    while datasets_helper.next_dataset():
        log_writer.add_log("Starting testing dataset {}".format(datasets_helper.get_dataset_name()))
        tokenizer = Tokenizer(num_words=num_of_words)
        generator = datasets_helper.text_generator()
        log_writer.add_log("Starting preprocessing and tokenization.")
        tokenizer.fit_on_texts(generator)
        log_writer.add_log("Done. Building model now.")

        batch_size = 256
        gauss_noise = 0.5
        epochs = 1
        val_split = 0.2
        val_data_count = int(datasets_helper.get_num_of_train_texts() * val_split)

        model = Sequential()
        model.add(Embedding(num_of_words, embedding_dim))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(datasets_helper.get_num_of_topics(),activation='softmax'))

        log_writer.add_log("Compiling model")
        model.layers[0].set_weights([get_embedding_matrix(num_of_words,embedding_dim,tokenizer.word_index)])
        model.layers[0].trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        plot_model(model, log_writer.get_plot_path("", "model-graph"), show_shapes=True)
        log_writer.add_log("Done. Now lets get training.")
        print(model.summary())
        #callbacks = [keras.callbacks.TensorBoard(log_dir=datasets_helper.get_tensor_board_path())]
        history = model.fit(
            x=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), batch_size, datasets_helper.get_num_of_train_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics(),max_len), epochs=num_of_epochs, validation_data=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), batch_size, validation_count, num_of_words, tokenizer, ";", datasets_helper.get_num_of_topics(),max_len,start_point=datasets_helper.get_num_of_train_texts()-validation_count))
        #history = model.fit(x_train,y_train, epochs=8,batch_size=256,validation_data=(x_validation,y_valitadio))
        result = model.evaluate(x=Training_Text_Generator_RNN_Embedding(datasets_helper.get_test_file_path(), batch_size, datasets_helper.get_num_of_test_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics(),max_len))# model.evaluate(test_sequences,test_labels)
        print(result)
        result.append(datasets_helper.get_dataset_name())
        #model.summary(print_fn=result.append)
        results.append(result)
        log_writer.add_log("Done. Finishing this dataset.")
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss {}'.format(datasets_helper.get_dataset_name()))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(log_writer.get_plot_path(datasets_helper.get_dataset_name(), "loss"))


        plt.clf()
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy {}'.format(datasets_helper.get_dataset_name()))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(log_writer.get_plot_path(datasets_helper.get_dataset_name(), "acc"))
        plt.clf()

        log_writer.add_log("Finished testing dataset {}".format(datasets_helper.get_dataset_name()))

        log_writer.write_2D_list("results", results)
        del tokenizer
    log_writer.end_logging()
    datasets_helper.reset_dataset_counter()