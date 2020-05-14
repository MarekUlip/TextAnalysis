import errno

from nltk import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
#import czech_lemmatizer
#import czech_stemmer
#from czech_stopwords import cz_stopwords
from lang_detector import detect_lang
from gensim.parsing import preprocessing
from settings import Settings
import csv
import glob

from preprocessor import czech_lemmatizer, czech_stemmer
from preprocessor.czech_stopwords import cz_stopwords

csv.field_size_limit(2**24) #(sys.maxsize) in 64 bit version sys.maxsize throws overflowError hence use of 2**24

#list of stopwords some are not contained in nltk list. Used to make better preprocess
#TODO remove duplicated stopwords
stp_wrds = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your']

class TextPreprocessor:
    def __init__(self, settings=None):
        if settings is None:
            self.settings_manager = Settings()
            self.settings = self.settings_manager.settings
        else:
            self.settings = settings
        self.lemma = WordNetLemmatizer()
        self.tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

    def reload_settings(self):
        """
        Forces reload of settings. If some changes were added after constructing this class this method will apply them.
        """
        self.settings_manager.load_settings()
        self.settings = self.settings_manager.settings

    def preprocess_texts(self):
        """
        Preprocesses and saved preprocessed texts onto disk (path must be provided via settings)
        """
        self.save_texts(self.prep_texts())

    def load_and_prep_csv(self, csvs, lang, load_for_test, col_to_take, delimeter=';', preproces_limit=False, preproces_limit_count=100):
        """
        Loads and preprocesses provided csv files
        :param csvs: list csv file paths to be loaded and preprocessed
        :param lang: language that should be used for preprocessing
        :param load_for_test: Defines form at which documents will be returned (True: document will be in form of tuple(topic index, string text),
                False: document will be in form of list of word strings)
        :param col_to_take: Defines which column in csv row should be preprocessed
        :param delimeter: Delimeter that was used to divide single items of row
        :param preproces_limit: Indicates wheter there is a limit to a number of preprocessed documents
        :param preproces_limit_count: if preproces_limit is set True then this number will be used to limit the number of preprocessed documents
        :return: List of preprocessed documents where a document form is defined by a parameter load_for_test
        """
        articles = []
        processed = 0
        if lang == "cz":
            prep_method = self.prep_text_czech
        else:
            prep_method = self.prep_text_eng
        for item in csvs:
            with open(item, encoding='utf-8', errors='ignore') as csvfile:
                csv_read = csv.reader(csvfile, delimiter=delimeter)
                for row in csv_read:
                    if not row[col_to_take]:
                        print("Empty string skipping")
                        continue
                    if load_for_test:
                        articles.append((int(row[0]), prep_method(row[col_to_take])))
                    else:
                        articles.append(prep_method(row[col_to_take]).split())
                    if preproces_limit:
                        processed += 1
                        if processed >= preproces_limit_count:
                            break
        return articles

    @staticmethod
    def load_csv(csvs, delimeter=';'):
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
                    articles.append(row)
        return articles

    def prep_texts(self, docs=None, lang=None):
        """
        Preprocesses provided docs and returns them as list
        :param docs: list of documents to be preprocessed if no list is provided csv file from settings path is loaded instead
        :param lang: Languange to be used with preprocessing. If no language is specified language is guessed based on number of stopwords.
        :return: list of preprocessed documents
        """
        if docs is None:
            docs = self.get_texts()
        clean_docs = []
        if lang is None:
            lang = detect_lang(docs[0])
            print("Language detected as "+lang)
        for doc in docs:
            if lang == "cz":
                clean_docs.append(self.prep_text_czech(doc))
            else:
                clean_docs.append(self.prep_text_eng(doc))
        return clean_docs

    def prep_texts_by_one(self, path=None, lang=None):
        """
        Preprocesses text files in a provided folders. Files are rewritten with their preprocessed version.
        :param path: path to text files to be preprocessed if no list is provided csv file from settings path is loaded instead
        :param lang: Languange to be used with preprocessing. If no language is specified language is guessed based on number of stopwords.
        """
        if path is None:
            path = self.settings["file_folder"]
        path += "/*.txt"
        files = glob.glob(path)
        for name in files:
            try:
                with open(name, 'r+', encoding="utf8") as f:
                    text = f.read().replace('\n', '')
                    f.seek(0)
                    f.truncate()
                    if lang is None:
                        lang = detect_lang(text)
                    if lang == "cz":
                        f.write(self.prep_text_czech(text))
                    else:
                        f.write(self.prep_text_eng(text))

            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise
        return [] #empty list for compatability reasons

    def save_texts(self, docs):
        """
        Saves provided texts into a file folder set in settings
        :param docs: documents to be saved
        """
        path = self.settings["file_folder"]
        path += "/*.txt"
        files = glob.glob(path)
        for index, name in enumerate(files):
            try:
                with open(name, 'w', encoding="utf8") as f:
                    f.write(" ".join(word for word in docs[index]))

            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise

    def get_texts(self, path=None):
        """
        Loads documents from provided path
        :param path: path from which documents will be loaded. If no path is specified path set via settings will be used instead.
        :return: list containing documents strings
        """
        if path is None:
            path = self.settings["file_folder"]
        docs = []
        path += "/*.txt"
        files = glob.glob(path)
        for name in files:
            try:
                with open(name, 'r', encoding="utf8") as f:
                    docs.append(f.read().replace('\n', ''))

            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise
        return docs

    def get_text(self, path):
        """
        Loads single text file from provided path
        :param path: path from which document will be loaded
        :return: loaded document string not processed
        """
        if path is None:
            print("No path specified returning")
            return ""
        doc = None
        file = glob.glob(path)
        try:
            with open(file[0], 'r', encoding="utf8") as f:
                doc = f.read().replace('\n', '')

        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
        return doc

    def prep_text_czech(self, text):
        res = preprocessing.strip_punctuation(text.lower())
        if self.settings['strip_nums']:
            res = preprocessing.strip_numeric(res)

        if self.settings['use_lemmatizer']:
            res = " ".join([czech_lemmatizer.lemmatize(word) for word in res.split()])

        res = " ".join([word for word in res.split() if word not in cz_stopwords])

        if self.settings['strip_short']:
            res = preprocessing.strip_short(res, minsize=3)
        if self.settings['use_stemmer']:
            res = " ".join([czech_stemmer.cz_stem(word) for word in res.split()])
        return res

    def prep_text_eng(self, text):
        # TODO optimalize
        res = preprocessing.strip_punctuation(text.lower())
        if self.settings['strip_nums']:
            res = preprocessing.strip_numeric(res)

        if self.settings['use_lemmatizer']: #TODO careful with using lemmatizer before removing stop words (performance)
            #res = " ".join([self.lemma.lemmatize(word, self.get_wordnet_pos(word)) for word in res.split() if len(word) > 2])
            res = " ".join(
                [self.lemma.lemmatize(word) for word in res.split()])

        if self.settings['remove_stop_words']:
            res = preprocessing.remove_stopwords(res)
            res = " ".join(word for word in res.split() if word not in stp_wrds)


        if self.settings['strip_short']:
            res = preprocessing.strip_short(res, minsize=3)

        if self.settings['use_stemmer']:
            res = preprocessing.stem_text(res)
        # normalized = " ".join(lemma.lemmatize(word) for word in res.split())
        return res

    def get_wordnet_pos(self, word):
        """
        Returns Part Of Speech tag for a provided word. Used with lemmatizer. Note that using this method makes lemmatization
        very slow
        :param word: word for which POS tag should be found
        :return: POS tag
        """
        """Map POS tag to first character lemmatize() accepts"""

        return self.tag_dict.get(nltk.pos_tag([word])[0][1][0].upper(), wordnet.NOUN)


#TextPreprocessor().prep_texts_by_one()
