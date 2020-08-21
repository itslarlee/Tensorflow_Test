import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data.head())

# tambien se llama label

predict = "G3"

# Se va a quitar de la lista lo que uno quiere predecir
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# Se va a dividir en 4 arrays para dividirse en data para que aprenda y data diferente para el testing
"""
best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # Reminder for Lee: Estudia las librerias de numpy y sklearn

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


# Estudiar que significa los datos de aqui abajo(Co e Intercept)
print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("Datos: ", x_test[x], " ||| Prediccion: ", predictions[x], " ||| Valor real: ", y_test[x])

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()