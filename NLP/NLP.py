from sklearn.feature_extraction.text import CountVectorizer
from tnize import GetTokens

corpus = ['Съешь еще этих мягких булок, да выпей чаю']
vectorizer = CountVectorizer(tokenizer=GetTokens)
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())