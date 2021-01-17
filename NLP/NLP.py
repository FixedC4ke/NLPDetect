#основной файл проекта, выводит вероятности, с которыми фраза принадлежит к различным категориям
#для обучения модели с нуля необходимо запускать learn.py
from joblib import load, dump
from scipy import sparse
from pandas import read_csv as read
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as train
import numpy as np



csv = read('myds.csv', delimiter=';', encoding='ansi')

catnames = {1: "расизм", 2: 'сексизм', 3: 'призыв к насилию', 4:'распространение наркотиков', 5:'материалы для взрослых', 6:'спам', 0:'допустимые сообщения'}

vw = load('vectorw.joblib')
vc = load('vectorc.joblib')
model = load('model.joblib')

le = LabelEncoder()
categories = le.fit_transform(csv.Category)

text = [input("Введите сообщение: ")]

vectorw = vw.fit_transform(text)
vectorc = vc.fit_transform(text)


data = sparse.hstack([vectorw, vectorc])

classnames = le.inverse_transform(model.classes_).tolist()
probs = model.predict_proba(data).tolist()[0]

predictedclass = model.predict(data)

print('[Результат] Введенное сообщение относится к категории "'+catnames[le.inverse_transform(predictedclass).tolist()[0]]+'"')
print('\nВероятности попадания сообщения в другие группы:')
for i in range(0, len(probs)):
    print(catnames[classnames[i]] + ": " + "{:.0%}".format(probs[i]))

choice = input("\nБыло ли предсказание верным? (y/n): ")
if choice == 'n':
    cat = int(input("Введите верную категорию (1: 'расизм', 2: 'сексизм', 3: 'призыв к насилию', 4:'распространение наркотиков', 5:'материалы для взрослых', 6:'спам', 0:'допустимые сообщения'): "))
    y = le.transform([cat])
    model.partial_fit(data, y)
elif choice == 'y':
    print(predictedclass)
    model.partial_fit(data, predictedclass)

probs = model.predict_proba(data).tolist()[0]
print('\nПересчитана вероятность попадания сообщения в группы:')
for i in range(0, len(probs)):
    print(catnames[classnames[i]] + ": " + "{:.0%}".format(probs[i]))

choice = input("Применить изменения к модели? (y/n): ")
if choice == 'y':
    dump(model, 'model.joblib')