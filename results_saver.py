import datetime
import csv
import os
import matplotlib.pyplot as plt
import json
import time

class LogWriter:
    def __init__(self, log_file_path=None,log_file_desc="",base_time =str(int(round(time.time()) * 1000))):
        """
        :param log_file_path: path to a file into which logs will be written
        """
        if log_file_path is None:
            self.path = os.getcwd()+"\\results\\"+base_time+"{}\\".format(log_file_desc)
        else:
            self.path = log_file_path
        self.logs = ["*****************\n"]

    def add_log(self, log):
        """
        Adds on line into log file. Note that this method does not write to file. Use append_to_file to perform file writting
        :param log: Text to be appended
        """
        log = str(datetime.datetime.now()) + ": "+log+"\n"
        print(log)
        self.logs.append(log)
        if len(self.logs) > 10:
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

    def write_2D_list(self, list_name, statistics):
        """
        Writes provided 2D list into csv file
        :param list_name: List name used in file creation
        :param statistics: 2D list where each row represents one csv file line
        """
        filename = self.path + list_name + ".csv"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w+', newline='') as list_file:
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
        path = os.getcwd()+file_name+".png"
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
        filename = os.getcwd() + file_name + ".txt"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w+', newline='') as params_file:
            #params_writer = csv.writer(params_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            params_file.write(json.dumps(params_to_save))

    def get_plot_path(self, dataset_name,plot_name):
        path = self.path + dataset_name+"\\"+ plot_name + ".png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def apped_to_desc(self, desc):
        self.path += desc
