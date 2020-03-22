import majka

lemmatizer = None

def init_lemmatizer(majka_path):
    lemmatizer = majka.Majka('{}\\{}'.format(majka_path,'majka.w-lt'))
    lemmatizer.first_only = True
    lemmatizer.tags = False
    lemmatizer.negative = 'ne'


def lemmatize(word, majka_path):
    if lemmatizer is None:
        init_lemmatizer(majka_path)
    lemma = lemmatizer.find(word)
    if len(lemma) > 0:
        return lemma[0]['lemma']
    else:
        return word
