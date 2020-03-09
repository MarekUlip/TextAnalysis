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
# list created with txt that was from GitHub repository https://github.com/stopwords-iso/stopwords-cs under MIT license https://github.com/stopwords-iso/stopwords-cs/blob/master/LICENSE
cz_stopwords = ['a', 'aby', 'ahoj', 'aj', 'ale', 'anebo', 'ani', 'aniž', 'ano', 'asi', 'aspoåˆ', 'aspoň', 'atd', 'atp', 'az', 'aä\x8dkoli', 'ačkoli', 'až', 'bez', 'beze', 'blã\xadzko', 'blízko', 'bohuå¾el', 'bohužel', 'brzo', 'bude', 'budem', 'budeme', 'budes', 'budete', 'budeå¡', 'budeš', 'budou', 'budu', 'by', 'byl', 'byla', 'byli', 'bylo', 'byly', 'bys', 'byt', 'bä›hem', 'být', 'během', 'chce', 'chceme', 'chcete', 'chceå¡', 'chceš', 'chci', 'chtã\xadt', 'chtä›jã\xad', 'chtít', 'chtějí', "chut'", 'chuti', 'ci', 'clanek', 'clanku', 'clanky', 'co', 'coz', 'což', 'cz', 'daleko', 'dalsi', 'další', 'den', 'deset', 'design', 'devatenáct', 'devatenã¡ct', 'devä›t', 'devět', 'dnes', 'do', 'dobrã½', 'dobrý', 'docela', 'dva', 'dvacet', 'dvanáct', 'dvanã¡ct', 'dvä›', 'dvě', 'dál', 'dále', 'dã¡l', 'dã¡le', 'dä›kovat', 'dä›kujeme', 'dä›kuji', 'děkovat', 'děkujeme', 'děkuji', 'email', 'ho', 'hodnä›', 'hodně', 'i', 'jak', 'jakmile', 'jako', 'jakož', 'jde', 'je', 'jeden', 'jedenáct', 'jedenã¡ct', 'jedna', 'jedno', 'jednou', 'jedou', 'jeho', 'jehož', 'jej', 'jeji', 'jejich', 'jejã\xad', 'její', 'jelikož', 'jemu', 'jen', 'jenom', 'jenž', 'jeste', 'jestli', 'jestliå¾e', 'jestliže', 'jeå¡tä›', 'ještě', 'jež', 'ji', 'jich', 'jimi', 'jinak', 'jine', 'jiné', 'jiz', 'již', 'jsem', 'jses', 'jseš', 'jsi', 'jsme', 'jsou', 'jste', 'já', 'jã¡', 'jã\xad', 'jã\xadm', 'jí', 'jím', 'jíž', 'jšte', 'k', 'kam', 'každý', 'kde', 'kdo', 'kdy', 'kdyz', 'kdyå¾', 'když', 'ke', 'kolik', 'kromä›', 'kromě', 'ktera', 'ktere', 'kteri', 'kterou', 'ktery', 'která', 'kterã¡', 'kterã©', 'kterã½', 'které', 'který', 'kteå™ã\xad', 'kteři', 'kteří', 'ku', 'kvå¯li', 'kvůli', 'ma', 'majã\xad', 'mají', 'mate', 'me', 'mezi', 'mi', 'mit', 'mne', 'mnou', 'mnä›', 'mně', 'moc', 'mohl', 'mohou', 'moje', 'moji', 'moå¾nã¡', 'možná', 'muj', 'musã\xad', 'musí', 'muze', 'my', 'má', 'málo', 'mám', 'máme', 'máte', 'máš', 'mã¡', 'mã¡lo', 'mã¡m', 'mã¡me', 'mã¡te', 'mã¡å¡', 'mã©', 'mã\xad', 'mã\xadt', 'mä›', 'må¯j', 'må¯å¾e', 'mé', 'mí', 'mít', 'mě', 'můj', 'může', 'na', 'nad', 'nade', 'nam', 'napiste', 'napište', 'naproti', 'nas', 'nasi', 'naå¡e', 'naå¡i', 'načež', 'naše', 'naši', 'ne', 'nebo', 'nebyl', 'nebyla', 'nebyli', 'nebyly', 'nechť', 'nedä›lajã\xad', 'nedä›lã¡', 'nedä›lã¡m', 'nedä›lã¡me', 'nedä›lã¡te', 'nedä›lã¡å¡', 'nedělají', 'nedělá', 'nedělám', 'neděláme', 'neděláte', 'neděláš', 'neg', 'nejsi', 'nejsou', 'nemajã\xad', 'nemají', 'nemáme', 'nemáte', 'nemã¡me', 'nemã¡te', 'nemä›l', 'neměl', 'neni', 'nenã\xad', 'není', 'nestaä\x8dã\xad', 'nestačí', 'nevadã\xad', 'nevadí', 'nez', 'neå¾', 'než', 'nic', 'nich', 'nimi', 'nove', 'novy', 'nové', 'nový', 'nula', 'ná', 'nám', 'námi', 'nás', 'náš', 'nã¡m', 'nã¡mi', 'nã¡s', 'nã¡å¡', 'nã\xadm', 'nä›', 'nä›co', 'nä›jak', 'nä›kde', 'nä›kdo', 'nä›mu', 'ní', 'ním', 'ně', 'něco', 'nějak', 'někde', 'někdo', 'němu', 'němuž', 'o', 'od', 'ode', 'on', 'ona', 'oni', 'ono', 'ony', 'osm', 'osmnáct', 'osmnã¡ct', 'pak', 'patnáct', 'patnã¡ct', 'po', 'pod', 'podle', 'pokud', 'potom', 'pouze', 'pozdä›', 'pozdě', 'poå™ã¡d', 'pořád', 'prave', 'pravé', 'pred', 'pres', 'pri', 'pro', 'proc', 'prostä›', 'prostě', 'prosã\xadm', 'prosím', 'proti', 'proto', 'protoze', 'protoå¾e', 'protože', 'proä\x8d', 'proč', 'prvni', 'první', 'práve', 'pta', 'pä›t', 'på™ed', 'på™es', 'på™ese', 'pět', 'před', 'přede', 'přes', 'přese', 'při', 'přičemž', 're', 'rovnä›', 'rovně', 's', 'se', 'sedm', 'sedmnáct', 'sedmnã¡ct', 'si', 'sice', 'skoro', 'smã\xad', 'smä›jã\xad', 'smí', 'smějí', 'snad', 'spolu', 'sta', 'sto', 'strana', 'stã©', 'sté', 'sve', 'svych', 'svym', 'svymi', 'své', 'svých', 'svým', 'svými', 'svůj', 'ta', 'tady', 'tak', 'take', 'takhle', 'taky', 'takze', 'také', 'takže', 'tam', 'tamhle', 'tamhleto', 'tamto', 'tato', 'te', 'tebe', 'tebou', "ted'", 'tedy', 'tema', 'ten', 'tento', 'teto', 'ti', 'tim', 'timto', 'tipy', 'tisã\xadc', 'tisã\xadce', 'tisíc', 'tisíce', 'to', 'tobä›', 'tobě', 'tohle', 'toho', 'tohoto', 'tom', 'tomto', 'tomu', 'tomuto', 'toto', 'troå¡ku', 'trošku', 'tu', 'tuto', 'tvoje', 'tvá', 'tvã¡', 'tvã©', 'två¯j', 'tvé', 'tvůj', 'ty', 'tyto', 'tä›', 'tå™eba', 'tå™i', 'tå™inã¡ct', 'téma', 'této', 'tím', 'tímto', 'tě', 'těm', 'těma', 'těmu', 'třeba', 'tři', 'třináct', 'u', 'urä\x8ditä›', 'určitě', 'uz', 'uå¾', 'už', 'v', 'vam', 'vas', 'vase', 'vaå¡e', 'vaå¡i', 'vaše', 'vaši', 've', 'vedle', 'veä\x8der', 'večer', 'vice', 'vlastnä›', 'vlastně', 'vsak', 'vy', 'vám', 'vámi', 'vás', 'váš', 'vã¡m', 'vã¡mi', 'vã¡s', 'vã¡å¡', 'vå¡echno', 'vå¡ichni', 'vå¯bec', 'vå¾dy', 'více', 'však', 'všechen', 'všechno', 'všichni', 'vůbec', 'vždy', 'z', 'za', 'zatã\xadmco', 'zatímco', 'zaä\x8d', 'zač', 'zda', 'zde', 'ze', 'zpet', 'zpravy', 'zprávy', 'zpět', 'ä\x8dau', 'ä\x8dtrnã¡ct', 'ä\x8dtyå™i', 'å¡est', 'å¡estnã¡ct', 'å¾e', 'čau', 'či', 'článek', 'článku', 'články', 'čtrnáct', 'čtyři', 'šest', 'šestnáct', 'že']
stp_wrds = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your']
base_path = os.getcwd()
dataset_folder = base_path + "\\datasets\\"

train_file_name = "new-train"
test_file_name = "new-test"
skippable_datasets = [0,1,3,4,5,6,7,8,9]#[0,1,2,4,5,6,7,8]#[1,3,4,5,6,7]#[1,3,4,5,6,7,8]#[1,4,5,6,7]#[1,4,5]#
wanted_datasets = range(10)

def preprocess_sentence(sentence):
    #sentence = preprocessing.strip_short(sentence, 3)
    #print(sentence)
    sentence = sentence.lower()
    sentence = " ".join(preprocessing.preprocess_string(sentence,[preprocessing.strip_punctuation,preprocessing.strip_multiple_whitespaces, preprocessing.strip_numeric, preprocessing.strip_short]))
    sentence = " ".join(word for word in sentence.split() if word not in stp_wrds)
    #print()
    #print(sentence)
    #sentence = preprocessing.stem_text(sentence)
    return sentence

def preprocess_sentence_eng(sentence):
    sentence = sentence.lower()
    sentence = " ".join(preprocessing.preprocess_string(sentence, [preprocessing.strip_punctuation,
                                                                   preprocessing.strip_multiple_whitespaces,
                                                                   preprocessing.strip_numeric,
                                                                   preprocessing.strip_short]))
    sentence = " ".join(word for word in sentence.split() if word not in stp_wrds)
    return sentence

def preprocess_sentence_cz(sentence):
    sentence = sentence.lower()
    sentence = " ".join(word for word in sentence.split() if word not in cz_stopwords)
    return sentence

class Dataset_Helper():
    def __init__(self,preprocess):
        self.dataset_position = -1
        self.dataset_info = []
        self.load_dataset_info()
        self.current_dataset = None
        self.csv_train_file_stream = None
        self.preprocess = preprocess
        self.vectorized_labels = False
        self.preprocess_func = preprocess_sentence
        self.wanted_datasets = range(len(self.dataset_info)) #:list of dataset indexes to be analysed. Defaultly all indexes from file info.csv will be analysed

    def load_dataset_info(self):
        with open(dataset_folder+"info.csv",encoding="utf-8", errors="ignore") as settings_file:
            csv_reader = csv.reader(settings_file, delimiter=';')
            for row in csv_reader:
                self.dataset_info.append(row)

    def set_preprocess_function(self,lang):
        if lang == 'cz':
            self.preprocess_func = preprocess_sentence_cz
        elif lang == 'eng':
            self.preprocess_func = preprocess_sentence_eng

    def change_dataset(self,index):
        self.current_dataset = self.dataset_info[index]
        self.set_preprocess_function(self.dataset_info[index][5])
        self.vectorized_labels = int(self.dataset_info[index][6]) == 1
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

    def get_dataset_topic_names(self,file_name='topic-names'):
        topic_names = []
        with open(self.get_dataset_folder_path()+"{}.csv".format(file_name),encoding="utf-8", errors="ignore") as settings_file:
            csv_reader = csv.reader(settings_file, delimiter=';')
            for row in csv_reader:
                topic_names.append(row[1])
        return topic_names

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
                if self.vectorized_labels:
                    labels.append(s[0].split(','))
                else:
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
