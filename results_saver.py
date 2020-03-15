import datetime
import csv
import os
import matplotlib.pyplot as plt
import json
import time
import pandas as pd

from sklearn.metrics import confusion_matrix
import numpy as np

from neural_networks.aliaser import plot_model
from dataset_loader.dataset_helper import Dataset_Helper,get_root_folder

class LogWriter:
    def __init__(self, log_file_path=None,log_file_desc="",base_time =str(int(round(time.time()) * 1000)),result_desc='Neural'):
        """
        :param log_file_path: path to a file into which logs will be written
        """
        if log_file_path is None:
            self.path = get_root_folder()+"\\results\\{}\\".format(result_desc)+base_time+"{}\\".format(log_file_desc)
        else:
            self.path = log_file_path
        self.logs = ["*****************\n"]

    def add_log(self, log, save_now=False):
        """
        Adds on line into log file. Note that this method does not write to file. Use append_to_file to perform file writting
        :param log: Text to be appended
        """
        log = str(datetime.datetime.now()) + ": "+log+"\n"
        print(log)
        self.logs.append(log)
        if len(self.logs) > 10 or save_now:
            self.append_to_logfile()

    def append_to_logfile(self):
        """
        Appends all items in log memory into the log file and clears outputed lines from memory.
        """
        with open(self.path+"log.txt", "a+") as f:
            for item in self.logs:
                f.write(item)
            self.logs.clear()

    def end_logging(self):
        """
        Writes the rest of in memory logs. Currently equivalent to append_to_logfile.
        """
        self.append_to_logfile()

    def write_2D_list(self, list_name, statistics,write_mode='w+'):
        """
        Writes provided 2D list into csv file
        :param list_name: List name used in file creation
        :param statistics: 2D list where each row represents one csv file line
        """
        filename = self.path + list_name + ".csv"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode=write_mode, newline='') as list_file:
            list_writer = csv.writer(list_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for item in statistics:
                list_writer.writerow(item)

    def add_to_plot(self, line_name, points):
        """
        Adds provided points into plot in form of line. This method does not save into file
        Call draw_plot to save the chart.
        :param line_name: Line label that will be displayed in legend
        :param points: list of floats
        """
        points = [x * 100 for x in points]
        plt.plot(points, label=line_name)

    def draw_plot(self, plot_name, file_name, num_of_tests):
        """
        Saves added lines into png chart file and prepares plot for new line addition.
        :param plot_name: Chart name that will be displayed in img
        :param file_name: Name of a file to which should this chart be saved
        :param num_of_tests: Number of performed tests
        """
        plt.axis([0, num_of_tests, 0, 100])
        plt.title(plot_name)
        plt.xlabel("Číslo testu")
        plt.ylabel("Přesnost (%)")
        plt.legend()
        path = self.path+file_name+".png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.clf()

    def write_model_params(self, file_name, params):
        """
        Writes params of the tested model into a file
        :param file_name: path to a file into which the params should be written. If file does not exist it will be created.
        :param params: Paramas to be written into a file
        """
        params_to_save = {}
        for key, value in params.items():
            params_to_save[key.name] = value
        filename = self.path + file_name + ".txt"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w+', newline='') as params_file:
            #params_writer = csv.writer(params_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            params_file.write(json.dumps(params_to_save))

    def write_any(self,file_name,to_safe,write_mode,is_json=False):
        filename = self.path + file_name + ".txt"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if not is_json:
            to_safe = pd.DataFrame(to_safe).to_json()
        with open(filename, mode=write_mode, newline='') as params_file:
            params_file.write(to_safe+'\n')

    def get_plot_path(self, dataset_name,plot_name):
        return self.convert_name_to_file_path(dataset_name,plot_name,'.png')

    def convert_name_to_file_path(self,dataset_name, name, file_type=''):
        path = self.path + dataset_name + "\\" + name + file_type
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


    def apped_to_desc(self, desc):
        self.path += desc




def finish_dataset(model, gnr, dataset_helper: Dataset_Helper, log_writer: LogWriter, history):
    log_writer.write_any('model', model.to_json(), 'w+', True)
    plot_model(model, log_writer.get_plot_path("", "model-graph"), show_shapes=True)
    model.save_weights(log_writer.convert_name_to_file_path(dataset_helper.get_dataset_name(),'weights','.h5'))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss {}'.format(dataset_helper.get_dataset_name()))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(log_writer.get_plot_path(dataset_helper.get_dataset_name(), "loss"))
    plt.clf()

    if not dataset_helper.vectorized_labels:
        predicts = model.predict(x=gnr)
        predicts = predicts.argmax(axis=-1)
        labels = gnr.labels[:len(predicts)]  # datasets_helper.get_labels(datasets_helper.get_test_file_path())
        # print(confusion_matrix(labels[:len(predicts)],predicts))

        cm = confusion_matrix(labels, predicts)
        # print(cm)
        fig = plt.figure(figsize=(dataset_helper.get_num_of_topics(), dataset_helper.get_num_of_topics()))
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
            # bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        """topic_names = datasets_helper.get_dataset_topic_names()
        ax.set_xticklabels([''] + topic_names)
        ax.set_yticklabels([''] + topic_names)"""
        plt.savefig(log_writer.get_plot_path(dataset_helper.get_dataset_name(), 'confusion_matrix'))
        plt.clf()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy {}'.format(dataset_helper.get_dataset_name()))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(log_writer.get_plot_path(dataset_helper.get_dataset_name(), "acc"))
    plt.clf()