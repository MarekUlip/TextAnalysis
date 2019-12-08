from aliaser import *
import matplotlib.pyplot as plt
from training_text_generator_RNN_embedding import Training_Text_Generator_RNN_Embedding
from helper_functions import Dataset_Helper
from results_saver import LogWriter
from embedding_loader import get_embedding_matrix
import os
import sys


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


datasets_helper = Dataset_Helper(preprocess=False)

num_of_words = 15000
#embedding_dim = 50
max_len = 300
for embedding_dim in [100]:
    results_saver = LogWriter(log_file_desc="RNN-GloVe{}-preprocessing-more-neurons-epochs".format(embedding_dim))
    results = []

    while datasets_helper.next_dataset():
        results_saver.add_log("Starting testing dataset {}".format(datasets_helper.get_dataset_name()))
        validation_count = 500#datasets_helper.get_num_of_train_texts() // 10
        tokenizer = Tokenizer(num_words=num_of_words,
                             filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                             lower=False, split=' ')
        generator = datasets_helper.text_generator()
        results_saver.add_log("Starting preprocessing and tokenization.")
        tokenizer.fit_on_texts(generator)
        results_saver.add_log("Done. Building model now.")

        model = Sequential()
        enhanced_num_of_topics = 256#int(np.ceil(datasets_helper.get_num_of_topics())*2.5) #-datasets_helper.get_num_of_topics()/2))
        model.add(Embedding(num_of_words, embedding_dim))
        model.add(LSTM(enhanced_num_of_topics, return_sequences=True))
        #model.add(LSTM(enhanced_num_of_topics, return_sequences=True))
        #model.add(Bidirectional(LSTM(enhanced_num_of_topics, activation='relu', return_sequences=True)))
        model.add(LSTM(enhanced_num_of_topics))

        """model.add(Flatten())
        model.add(Dense(enhanced_num_of_topics, activation='relu'))
        model.add(Dense(enhanced_num_of_topics, activation='relu'))
        model.add(Dense(enhanced_num_of_topics, activation='relu'))"""
        #model.add(LSTM(40,activation='relu'))
        #model.add(Dense(enhanced_num_of_topics, activation='relu', input_shape=(num_of_words,)))
        #model.add(Dense(enhanced_num_of_topics, activation='relu'))
        model.add(Dense(datasets_helper.get_num_of_topics(),activation='softmax'))

        results_saver.add_log("Compiling model")
        #model.layers[0].set_weights([get_embedding_matrix(num_of_words,embedding_dim,tokenizer.word_index)])
        model.layers[0].trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        plot_model(model,results_saver.get_plot_path("","model-graph"),show_shapes=True)
        results_saver.add_log("Done. Now lets get training.")
        print(model.summary())
        batch_size = 128
        num_of_epochs = 30
        #callbacks = [keras.callbacks.TensorBoard(log_dir=datasets_helper.get_tensor_board_path())]
        history = model.fit_generator( generator=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), batch_size, datasets_helper.get_num_of_train_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics(),max_len), epochs=num_of_epochs, validation_data=Training_Text_Generator_RNN_Embedding(datasets_helper.get_train_file_path(), batch_size, validation_count, num_of_words, tokenizer, ";", datasets_helper.get_num_of_topics(),max_len,start_point=datasets_helper.get_num_of_train_texts()-validation_count))
        #history = model.fit(x_train,y_train, epochs=8,batch_size=256,validation_data=(x_validation,y_valitadio))
        result = model.evaluate_generator(generator=Training_Text_Generator_RNN_Embedding(datasets_helper.get_test_file_path(), batch_size, datasets_helper.get_num_of_test_texts(), num_of_words, tokenizer, ";",datasets_helper.get_num_of_topics(),max_len))# model.evaluate(test_sequences,test_labels)
        print(result)
        result.append(datasets_helper.get_dataset_name())
        #model.summary(print_fn=result.append)
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
        del tokenizer
    results_saver.end_logging()
    datasets_helper.reset_dataset_counter()