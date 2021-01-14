#в данном модуле содержится функция токенизации текста

import nltk
import string
from nltk.corpus import stopwords
import morph

def GetTokens(text):
    #nltk.download('punkt') #загрузка словаря со знаками препинания
    #nltk.download('stopwords')
    stop_words = stopwords.words("russian")
    tokens = nltk.word_tokenize(text.lower())
    tokens = [i for i in tokens if (i not in string.punctuation) and (i not in stop_words)]
    morphd = []
    for token in tokens:
        morphd.append(morph.morpholize(token))
    return morphd