from sklearn import svm
from sklearn import datasets

digits = datasets.load_digits()

"""Estimador" estableciendo el valor de gamma manualmente
para encontrar buenos valores afinando la prediccion necesito otras herramientas
(validacion cruzada ej.)"""

clf = svm.SVC(gamma=0.001, C=100.)

"""conjunto de entramiento, usamos todas las imagenes del dataset menos la ultima,
esta la reservamos para la prediccion.
seleccionando un conjunto de entrenamiento del dataset [:-1]
producimos una nueva matriz sin el ultimo elemento del dataset (digits.data)"""
entrenamiento = clf.fit(digits.data[:-1], digits.target[:-1])

"""Prediciendo(conjunto de prueba). Determina que imagen del conjunto de entrenamiento 
coincide mejor con la ultima imagen"""
prediccion = clf.predict(digits.data[-1:])

print(prediccion)
