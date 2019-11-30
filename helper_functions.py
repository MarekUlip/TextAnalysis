import os
import sys
import csv
from gensim.parsing import preprocessing

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

stp_wrds = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your']
base_path = os.getcwd()
dataset_folder = base_path + "\\datasets\\"

train_file_name = "new-train"
test_file_name = "new-test"
skippable_datasets = [0,1,3,4,5,6,7,8]#[0,1,2,4,5,6,7,8]#[1,3,4,5,6,7]#[1,3,4,5,6,7,8]#[1,4,5,6,7]#[1,4,5]#
wanted_datasets = [3,8]

def preprocess_sentence(sentence):
    sentence = " ".join(word for word in sentence.split() if word not in stp_wrds)
    sentence = preprocessing.strip_short(sentence, 3)
    sentence = preprocessing.stem_text(sentence)
    return sentence

class Dataset_Helper():
    def __init__(self,preprocess):
        self.dataset_position = -1
        self.dataset_info = []
        self.load_dataset_info()
        self.current_dataset = None
        self.csv_train_file_stream = None
        self.preprocess = preprocess
        self.wanted_datasets = range(len(self.dataset_info)) #:list of dataset indexes to be analysed. Defaultly all indexes from file info.csv will be analysed

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

    def set_wanted_datasets(self, wanted_datasets):
        self.wanted_datasets = wanted_datasets

    def skip_selected_datasets(self, selected_datasets):
        """
        Skips provided datasets so they are not analysed.
        :param selected_datasets: indexes of datasets starting from 0 that should be ignored
        """
        self.wanted_datasets = [index for index in self.wanted_datasets if index not in selected_datasets]
    def next_dataset(self):
        self.dataset_position += 1
        while self.dataset_position not in self.wanted_datasets:
            self.dataset_position += 1
            if self.dataset_position >= len(self.dataset_info):
                return False
        if self.csv_train_file_stream is not None:
            self.csv_train_file_stream.close()
            self.csv_train_file_stream = None
        self.change_dataset(self.dataset_position)
        return True

    def get_texts_as_list(self, csv_file_stream=None):
        return list(self.text_generator(csv_file_stream))

    def reset_dataset_counter(self):
        self.dataset_position = -1

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

    def get_tensor_board_path(self):
        path = base_path + "\\tensorBoard\\"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def reset_file_stream(self):
        if self.csv_train_file_stream is not None:
            self.csv_train_file_stream.seek(0)

    def open_file_stream(self, path):
        return open(path, encoding="utf-8", errors="ignore")

    def get_labels(self, path):
        with self.open_file_stream(path) as csv_file_stream:
            labels = []
            for s in csv.reader(csv_file_stream, delimiter=';'):
                # print("getting item based on {}".format(item))
                labels.append(int(s[0]))
        return labels



    def text_generator(self, csv_file_stream=None):
        if csv_file_stream is None:
            csv_file_stream = self.csv_train_file_stream
        for s in csv.reader(csv_file_stream, delimiter=';'):
            # print("getting item based on {}".format(item))
            if self.preprocess:
                yield preprocess_sentence(s[1])
            else:
                yield s[1]

    def text_generator_b(self, csv_file_stream = None):
        if csv_file_stream is None:
            csv_file_stream = self.csv_train_file_stream
        for text in csv_file_stream:
            if text == "":
                break
            s = text.split(";")
            if len(s) <= 1:
                print('uups')
                continue
            if self.preprocess:
                yield preprocess_sentence(s[1])
            else:
                yield s[1]
