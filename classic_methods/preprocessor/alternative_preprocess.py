import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

stemmer = LancasterStemmer()


def strip_html(text):
    """
    Removes html tags from text
    :param text: text from which tags should be removed
    :return: text without html tags
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    """
    Removes square brackets and all text that is inside those brackets. Use it only if you know that text inside square
    brackets is not useful for topic analysis
    :param text: text to be cleared
    :return: text without square brackets
    """
    return re.sub('\[[^]]*\]', '', text)


def denoise_text(text):
    """
    Combines strip_html and remove_between_square_brackets
    :param text: text to be denoise
    :return: denoised string
    """
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def replace_contractions(text):
    """
    Replace contractions in provided text
    :param text:
    :return: text with replaced contractions
    """
    return contractions.fix(text)

def remove_non_ascii(words):
    """
    Remove non-ASCII characters from provided words
    :param words: list of word strings
    :return: list of word strings with only ASCII characters
    """
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def remove_non_ascii_single(word):
    return unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def to_lowercase_single(word):
    return word.lower()

def remove_punctuation_single(word):
    return re.sub(r'[^\w\s]', '', word)

def replace_numbers_single(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords_single(word):
    if word not in stopwords.words('english'):
        return word
    else:
        return ''

def stem_words_single(word):
    return stemmer.stem(word)

def normalize(words):
    new_words = []
    for word in words:
        word = remove_non_ascii_single(word)
        word = to_lowercase_single(word)
        word = remove_punctuation_single(word)
        word = remove_stopwords_single(word)
        new_words.append(stem_words_single(word))
    """words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words)"""
    return new_words

def preprocess(text):
    text = denoise_text(text)
    text = replace_contractions(text)
    return " ".join(normalize(text.split()))