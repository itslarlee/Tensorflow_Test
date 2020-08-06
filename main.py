import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data.head())

# tambien se llama label

predict = "G3"

# Se va a quitar de la lista lo que uno quiere predecir
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Se va a dividir en 4 arrays para dividirse en data para que aprenda y data diferente para el testing

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print(accuracy)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range (len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
