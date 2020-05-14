import majka

lemmatizer = majka.Majka('majka.w-lt')
lemmatizer.first_only = True
lemmatizer.tags = False
lemmatizer.negative = 'ne'


def lemmatize(word):
    lemma = lemmatizer.find(word)
    if len(lemma) > 0:
        return lemma[0]['lemma']
    else:
        return word
