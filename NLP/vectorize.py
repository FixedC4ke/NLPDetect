#векторизация с использованием TF-IDF

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tnize import GetTokens

def Vectorize(corpus):
    vectorizer = TfidfVectorizer(tokenizer=GetTokens)
    X = vectorizer.fit_transform(corpus)
    return X.toarray()