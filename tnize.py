#в данном модуле содержится функция токенизации текста

import nltk
import string
from nltk.corpus import stopwords
import morph
import difflib
from dictopen import GetDict

def GetClosestMatch(word, rusdict):
    res = difflib.get_close_matches(word, rusdict, n=1, cutoff=0.7)
    if len(res)>0:
        return res[0]
    else:
        return word

def GetTokens(text):
    rusdict = GetDict()
    stop_words = stopwords.words("russian")
    tokens = nltk.word_tokenize(text.lower())
    tokens = [i for i in tokens if (i not in string.punctuation) and (i not in stop_words)]
    morphd = []
    for token in tokens:
        nfword = morph.morpholize(token)
        morphd.append(GetClosestMatch(nfword, rusdict))
    return morphd