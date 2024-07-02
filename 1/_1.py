#Вариант 1 (классическая модель LinearRegression)
import csv
import numpy
import sklearn
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn import preprocessing

def regr(data):
    X = data[:, 1:-1]
    y = data[:, -1]
    res = 0
    for i in range(10):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.7)
        lr = linear_model.LinearRegression()
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test)
        n = 0
        for j in range(len(X_test)):
            if abs(y_test[j] - pred[j]) < 1:
                n += 1
        acc = n / len(X_test) * 100
        print(f'{ i + 1 }) Точность: { round(acc, 5) }')
        res += acc
    print(f'Cреднее значение качества регрессии: { round(res / 10, 5) }')

data = numpy.genfromtxt('..\..\winequalityN.csv', delimiter=',', skip_header=True)
type = []
N = 0
f = open('..\..\winequalityN.csv', "r")
for _ in f:
    rd = csv.reader(f, delimiter=',')
    for row in rd:
        type.append(row[0])
        if row[0] == 'white':
            N += 1
f.close()
le = preprocessing.LabelEncoder()
data = numpy.hstack((numpy.array([le.fit_transform(type)]).T, data))
si = SimpleImputer()
si.fit(data)
data = si.transform(data)
for i in range(len(data[0]) - 1):
    data[..., i] = preprocessing.normalize([data[..., i]])

print('Красное вино:')
regr(data[N:])

print('\nБелое вино:')
regr(data[:N])

print('\nОбщие данные:')
regr(data)