from helper_functions import preprocess_sentence
from gensim.corpora import Dictionary

class GensimTextGenerator:
    def __init__(self, csv_file_path, preprocess):
        self.csv_file_path = csv_file_path
        self.dictionary:Dictionary = None
        self.csv_file_stream = open(self.csv_file_path, encoding="utf-8", errors="ignore")
        self.preprocess = preprocess
        self.iterator = None

    def reset_file_stream(self):
        self.csv_file_stream.seek(0)

    def init_dictionary(self):
        self.dictionary = Dictionary(line.lower().split(';')[1].split() for line in open(self.csv_file_path, encoding="utf-8", errors="ignore"))#self.__text_generator_dict)
        self.reset_file_stream()

    def get_dictionary(self):
        return self.dictionary

    def get_iterator(self):
        return MyIterator(self.csv_file_stream,self.dictionary,self.preprocess)

    def get_corpus(self):
        corpus = []
        for text in self.csv_file_stream:
            #print("generating")
            if text == "":
                continue
            s = text.split(";")
            if len(s) <= 1:
                continue
            if self.preprocess:
                corpus.append(self.dictionary.doc2bow(preprocess_sentence(s[1]).split()))
            else:
                corpus.append(self.dictionary.doc2bow(s[1].split()))
        return corpus

    def __text_generator_dict(self):
        for text in self.csv_file_stream:
            #print("generating")
            if text == "":
                break
            s = text.split(";")
            if len(s) <= 1:
                break
            if self.preprocess:
                yield preprocess_sentence(s[1])
            else:
                yield s[1]

    def text_generator(self):
        for text in self.csv_file_stream:
            #print("generating")
            if text == "":
                break
            s = text.split(";")
            if len(s) <= 1:
                break
            if self.preprocess:
                yield self.dictionary.doc2bow(preprocess_sentence(s[1]).split())
            else:
                yield self.dictionary.doc2bow(s[1].split())

class MyIterator(object):
    def __init__(self, csv_file_stream, dictionary, preprocess):
        self.csv_file_stream = csv_file_stream
        self.dictionary = dictionary
        self.preprocess = preprocess
    def __iter__(self):
        for text in self.csv_file_stream:
            # print("generating")
            if text == "":
                break
            s = text.split(";")
            if len(s) <= 1:
                break
            if self.preprocess:
                yield self.dictionary.doc2bow(preprocess_sentence(s[1]).split())
            else:
                yield self.dictionary.doc2bow(s[1].split())