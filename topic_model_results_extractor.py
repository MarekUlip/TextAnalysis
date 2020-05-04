import os
import sys
from enum import Enum
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import csv
import pandas as pd
import warnings


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
root = tk.Tk()
root.withdraw()
def get_root_folder():
    path = Path(os.getcwd())

    return str(path)

def process_classic_results():
    directory = filedialog.askdirectory()
    all_results = []
    for file in os.listdir(directory):
        results = [[0 for i in range(4)] for i in range(2)]
        print(os.getcwd())
        try:
            filename = os.fsdecode(file)
            results[0][0] = '{}(in)'.format(filename)
            results[1][0] = '{}(out)'.format(filename)
            with open(directory + '/' +filename+ '/'+'log.txt', 'r', encoding='utf8') as log:
                for index, line in enumerate(log):
                    if index == 5:
                        print(line)
                        results[0][3] = line.split()[5]
                    if index == 12:
                        print(line)
                        results[1][3] = line.split()[5]
                    if index == 15:
                        results[0][1] = line.split()[7]
                        results[0][2] = line.split()[11]
                    if index == 20:
                        results[1][1] = line.split()[7]
                        results[1][2] = line.split()[11]
            all_results.extend(results)
        except Exception as e:

            warnings.warn(e)
            continue
        frame = pd.DataFrame(all_results)
        Path(os.getcwd() + '/compiled-results').mkdir(parents=True, exist_ok=True)
        frame.to_csv(os.getcwd() + '/compiled-results/neural-topic-model-results.csv')

process_classic_results()





