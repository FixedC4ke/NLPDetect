from vectorize import Vectorize
from pandas import read_csv as read
from sklearn.model_selection import train_test_split as train

corpus = ['Съеж ище этех мяхких булак да выпий чаю',
          'TF-IDF — это статистическая мера, которая позволяет определить наиболее важные слова для текста в корпусе с помощью двух параметров: частот слов в каждом документе и количества документов, содержащих определенное слово (более подробно здесь). Рассчитав для каждого слова в сообщении TF-IDF, получаем векторное представление этого сообщения. ']

data = read('myds.csv', delimiter=';', encoding='ansi')
print(data.head())

X = data.values[::, 1:]
Y = data.values[::, 0:1]
print(X)
print(Y)