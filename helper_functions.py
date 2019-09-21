import os
import sys
import csv

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
csv.field_size_limit(maxInt)

base_path = os.getcwd()
dataset_folder = base_path + "\\datasets\\"
train_file_name = "new-train"
test_file_name = "new-test"
skippable_datasets = [0,1,2,4,5,6]

class Dataset_Helper():
    def __init__(self):
        self.dataset_position = -1
        self.dataset_info = []
        self.load_dataset_info()
        self.current_dataset = None
        self.csv_train_file_stream = None

    def load_dataset_info(self):
        with open(dataset_folder+"info.csv",encoding="utf-8", errors="ignore") as settings_file:
            csv_reader = csv.reader(settings_file, delimiter=';')
            for row in csv_reader:
                self.dataset_info.append(row)

    def change_dataset(self,index):
        self.current_dataset = self.dataset_info[index]
        if self.csv_train_file_stream is not None:
            self.csv_train_file_stream.close()
            self.csv_train_file_stream = None
        self.csv_train_file_stream = open(self.get_train_file_path(), encoding="utf-8", errors="ignore")

    def next_dataset(self):
        self.dataset_position += 1
        while self.dataset_position in skippable_datasets:
            self.dataset_position += 1
        if self.dataset_position >= len(self.dataset_info):
            return False
        if self.csv_train_file_stream is not None:
            self.csv_train_file_stream.close()
            self.csv_train_file_stream = None
        self.change_dataset(self.dataset_position)
        return True

    def get_num_of_test_texts(self):
        return int(self.current_dataset[4])

    def check_dataset(self):
        if self.current_dataset is None:
            raise ValueError("No current dataset was set.")

    def get_num_of_train_texts(self):
        return int(self.current_dataset[3])

    def get_num_of_topics(self):
        return int(self.current_dataset[2])

    def get_dataset_name(self):
        return self.current_dataset[1]

    def get_dataset_folder_name(self):
        return self.current_dataset[0]

    def get_dataset_folder_path(self):
        return "{}{}\\".format(dataset_folder, self.get_dataset_folder_name())

    def get_test_file_path(self):
        return self.get_dataset_folder_path()+test_file_name+".csv"

    def get_train_file_path(self):
        return self.get_dataset_folder_path()+train_file_name+".csv"

    def text_generator(self):
        for text in self.csv_train_file_stream:
            if text == "":
                break
            s = text.split(";")
            if len(s) <= 1:
                break
            yield s[1]
