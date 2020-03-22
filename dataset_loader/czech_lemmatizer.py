import majka

lemmatizer = None

class CzechLemmatizer:
    def __init__(self, majka_path):
        self.lemmatizer = self.init_lemmatizer(majka_path)

    def init_lemmatizer(self, majka_path):
        lemmatizer = majka.Majka('{}\\{}'.format(majka_path,'majka.w-lt'))
        lemmatizer.first_only = True
        lemmatizer.tags = False
        lemmatizer.negative = 'ne'
        return lemmatizer


    def lemmatize(self, word):
        lemma = self.lemmatizer.find(word)
        if len(lemma) > 0:
            return lemma[0]['lemma']
        else:
            return word
