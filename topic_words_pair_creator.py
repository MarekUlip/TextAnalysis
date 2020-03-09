import csv

from helper_functions import Dataset_Helper
from lda_impl import Lda
#from NeuralLDAanalysisMethods import *

dataset_helper = Dataset_Helper(True)
dataset_helper.set_wanted_datasets([2])
dataset_helper.next_dataset()
num_of_topics = dataset_helper.get_num_of_topics()
documents = dataset_helper.get_texts_as_list()
model = Lda(num_of_topics,20,
            passes=25,
            iterations=25)

model.train(documents)
doc = documents[0]
res = max(model.analyse_text(doc), key=lambda item: item[1])[0]
print(model.model.print_topics())
print(doc)
#print(model.model.get_term_topics('sdfsfds',0.0))
pairs = []
for doc in documents:
    max_words = []
    for word in doc.split():
        topics = model.model.get_term_topics(word,0.0)
        if topics is None or len(topics)==0:
            continue
        highest = max(topics, key=lambda item: item[1])
        if highest[0] == res:
            max_words.append(word)
    pairs.append([doc," ".join(set(max_words))])
with open("topic_pairs.csv", mode='w', newline='', errors="ignore", encoding='utf8') as output:
    csv_writer = csv.writer(output, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for item in pairs:
        csv_writer.writerow(item)
