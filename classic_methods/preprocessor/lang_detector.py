from nltk.corpus import stopwords
from gensim.parsing import preprocessing
import czech_stopwords


def detect_lang(text):
    eng_stopwords = set(stopwords.words('english'))
    cz_stopwords = czech_stopwords.cz_stopwords
    text = prep_text(text)
    eng = 0
    cz = 0
    for word in text:
        if word in cz_stopwords:
            cz += 1
            continue
        if word in eng_stopwords:
            eng += 1
    print("cz "+str(cz))
    print(eng)
    if cz < eng:
        return "eng"
    else:
        return "cz"


def prep_text(text):
    text = preprocessing.strip_punctuation(text.lower())
    #Stripping short word due to similarity of some English and Czech short words (i.e. on, a etc.)
    text = preprocessing.strip_short(text, minsize=3)
    return set(text.split())
