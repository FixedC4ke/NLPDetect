#векторизация с использованием TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from tnize import GetTokens

def Vectorize(corpus):
    vector_words = TfidfVectorizer(tokenizer=GetTokens, analyzer='word', ngram_range=(1,3))
    vector_char = TfidfVectorizer(analyzer='char', lowercase=True, ngram_range=(3,6))
    x_words = vector_words.fit_transform(corpus)
    x_char = vector_char.fit_transform(corpus)
    return sparse.hstack([x_words, x_char])