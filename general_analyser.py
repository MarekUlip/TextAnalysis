import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_helper import Dataset_Helper
from results_saver import LogWriter
from models.Dense import DenseModel
import os
import sys
import time
from sklearn.metrics import confusion_matrix
from aliaser import plot_model,Tokenizer

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

datasets_helper = Dataset_Helper(preprocess=True)
num_of_words = 10000
models = [DenseModel(),DenseModel()]
batch_size = 128
base_time = str(int(round(time.time()) * 1000))
counter = 0
for model in models:
    results_saver = LogWriter(log_file_desc=model.get_description(),base_time=base_time+str(counter))
    results = []
    counter+=1
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
        model.set_base_params(datasets_helper.get_num_of_topics(),num_of_words)

        plot_model(model.get_compiled_model(), results_saver.get_plot_path("", "model-graph"), show_shapes=True)
        results_saver.add_log("Done. Now lets get training.")
        history = model.fit(datasets_helper=datasets_helper, batch_size=batch_size, tokenizer=tokenizer, validation_count=validation_count)
        result = model.evaluate(datasets_helper=datasets_helper, batch_size=batch_size, tokenizer=tokenizer)# model.evaluate(test_sequences,test_labels)
        print(result)
        result.append(datasets_helper.get_dataset_name())
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
    datasets_helper.reset_dataset_counter()