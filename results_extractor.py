import os
import sys
from enum import Enum
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import csv
import pandas as pd
import warnings
class NeuralType(Enum):
    DENSE = 0
    RNN = 2
    BIDI_GRU = 4
    CONV_GRU = 6
    GRU = 3
    CONV = 5
    EMBEDDING_GLOVE_LSTM = 7
    EMBEDDING_GLOVE_TRAINED_LSTM = 8
    EMBEDDING_TRAINED_LSTM = 9
    LSTM = 1

class ClassicType(Enum):
    LDA_Sklearn = 2
    LDA = 0
    LSA = 1
    NB = 3
    SVM = 4
    DT = 5
    RF = 6


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
root = tk.Tk()
root.withdraw()
#print(filedialog.askdirectory())
#log_writer = #simpledialog.a(title="Test Name", prompt="Insert test name:", initialvalue='LSTM_')
def build_names(name_type):
    names = []
    if 'neural' in name_type:
        for name in NeuralType:
            names.append(name.name)
    if 'classic' in name_type:
        for name in ClassicType:
            names.append(name.name)
    return names
def get_root_folder():
    path = Path(os.getcwd())

    return str(path)

def process_classic_results():
    names = build_names('classic')
    directory = filedialog.askdirectory()
    dataset_results = {}
    for file in os.listdir(directory):
        print(os.getcwd())
        try:
            filename = os.fsdecode(file)
            if 'no-prep' in filename:
                preprocess = 'no-prep'
            else:
                preprocess = 'prep'
            model_name = ''
            for name in names:
                if name in filename:
                    model_name = name
                    break
            results = []
            with open(directory + '/' + filename + '/stats.csv', 'r', encoding='utf8') as stats:
                csv_reader = csv.reader(stats)
                is_results = False
                is_dataset_name = False
                result = []
                for line in csv_reader:
                    print(line)
                    last_index = len(line) - 1
                    if last_index<0:
                        continue

                    if is_dataset_name:
                        result.append(line[0])
                        is_dataset_name = False
                        results.append(result)
                        result = []
                    if is_results:
                        result.append(float(line[last_index]))
                        is_results = False
                        is_dataset_name = True
                    if line[last_index] == 'Average':
                        is_results = True
            prep_method_name = preprocess
            for result in results:
                if result[1] not in dataset_results:
                    dataset_results[result[1]] = pd.DataFrame(index=names,
                                                              columns=['prep',
                                                                       'no-prep'])
                dataset_results[result[1]][prep_method_name][model_name] = result[0]
        except Exception as e:
            #warnings.warn(e)
            continue
    for key, value in dataset_results.items():
        Path(os.getcwd() + '/compiled-results').mkdir(parents=True, exist_ok=True)
        value.to_csv(os.getcwd() + '/compiled-results/classic-{}-results.csv'.format(key))

def process_neural_results():
    names = build_names('neural')
    directory = filedialog.askdirectory()
    dataset_results = {}
    for file in os.listdir(directory):
        print(os.getcwd())
        try:
            filename = os.fsdecode(file)
            if 'no-prep' in filename:
                preprocess = 'no-prep'
            else:
                preprocess = 'prep'
            model_name = ''
            for name in names:
                if name in filename:
                    model_name = name
                    break
            with open(directory+'/'+filename+'/log.txt','r',encoding='utf8') as log:
                content = log.read()
                if 'binary' in content:
                    tokenizer_method = 'binary'
                else:
                    tokenizer_method = 'tfidf'
            results = []
            with open(directory+'/'+filename+'/results.csv','r',encoding='utf8') as stats:
                csv_reader = csv.reader(stats)
                for line in csv_reader:
                    results.append([float(line[1]),line[2]])
            prep_method_name = tokenizer_method +'-'+ preprocess
            for result in results:
                if result[1] not in dataset_results:
                    dataset_results[result[1]] = pd.DataFrame(index=names,columns=['binary-prep','binary-no-prep','tfidf-prep','tfidf-no-prep'])
                #print(dataset_results[result[1]])
                dataset_results[result[1]][prep_method_name][model_name] = result[0]
            #print(dataset_results)
        except Exception as e:
            #warnings.warn(e)
            continue
    for key, value in dataset_results.items():
        Path(os.getcwd()+'/compiled-results').mkdir(parents=True, exist_ok=True)
        value.to_csv(os.getcwd()+'/compiled-results/{}-results.csv'.format(key))

process_neural_results()
#process_classic_results()





