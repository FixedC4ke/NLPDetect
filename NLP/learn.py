#обучение модели
from vectorize import Vectorize
from pandas import read_csv as read
from sklearn.model_selection import train_test_split as train
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from joblib import dump

def Learn():
    logreg = LogisticRegression(n_jobs=-1)

    data = read('myds1.csv', delimiter=';', encoding='ansi')
    print("Датасет загружен в память\nПриводим загруженные данные к необходимому виду...")

    X = Vectorize(data.Sentence)
    le = LabelEncoder()
    y = le.fit_transform(data.Category)
    print("Начинаем процесс обучения...")


    X_train, X_test, y_train, y_test = train(X, y)
    logreg.fit(X_train, y_train)
    print("Обучение закончено")
    dump(logreg, 'model.joblib')
    print("Дамп модели сохранен в файл 'model.joblib'")