import csv
import numpy as np
import os
import random

base_path = os.getcwd()
csv_folder = base_path + "\\datasets\\"
print(csv_folder)

def load_csv(csvs, delimeter=';', labels_separated = True, shuffle=False, rows_to_load=None):
    """
    Loads csv files and returns list of rows from all provided csvs. No preprocessing is done
    :param csvs: csv files to be loaded
    :param delimeter: Delimeter that was used to divide single items of row
    :return: ist of rows from all provided csvs
    """
    articles = []
    for item in csvs:
        with open(item, encoding='utf-8', errors='ignore') as csvfile:
            csv_read = csv.reader(csvfile, delimiter=delimeter)
            for row in csv_read:
                articles.append(row)#[int(row[0]),row[1]])
                if rows_to_load is not None:
                    if len(articles) >= rows_to_load:
                        break
        if labels_separated:
            if shuffle:
                random.shuffle(articles)
            articles = np.array(articles)
            return (articles[:5000,0].astype(int),articles[:5000,1])
        else:
            return articles