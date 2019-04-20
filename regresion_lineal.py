
#Prediccion de valor medio de una casa en base a cantidad de habitaciones

import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

boston = datasets.load_boston()

#informacion que hay en el dataset
#print(boston.keys())

#data, target, feature_names, DESCR, filename

#descripcion detallada del dataset
#print(boston.DESCR)

#forma del dataset
#print(boston.data.shape)
#506 elementos 13 atributos (columnas)

#Nombre de las etiquetas en las columnas
#print(boston.feature_names)
'''['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']'''
#RM = numero de habitaciones en la casa

#Prediccion con regresion lineal simple

# Y= W0 + W1 . X   
# dependiente = interseccion + pendiente . independiente
#               

#Mi variable independiente sera X=numero de habitaciones
#nueva array con los datos de mi X
x = boston.data[:, np.newaxis, 5]

y = boston.target

#grafica con los datos correspondientes

plt.scatter(x,y)
plt.xlabel("Num. habitaciones")
plt.ylabel("Valor medio")
plt.show()

#separo los datos para ENTRENAMIENTO de los datos de PRUEBA, armo dos conjuntos.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#Defino el tipo de algoritmo a usar, en este caso
# Regresion lineal

lr = linear_model.LinearRegression()

#Entreno a mi modelo
lr.fit(x_train, y_train)

#Realizo la prediccion con mis datos de prueba
y_pred = lr.predict(x_test)


#grafica del los datos del modelo

plt.scatter(x_test,y_test)
plt.plot(x_test, y_pred, color='red', linewidth=3)
plt.title('Modelo regresion lineal simple')
plt.xlabel('Num de habitaciones')
plt.ylabel('Valor medio')

plt.show()

#Calculo del valor de la interseccion y la pendiente

print("\nDatos del modelo\n")
print('Valor de la pendiente = '+str(lr.coef_)+"\n")
print('Valor de la interseccion = '+str(lr.intercept_))

#Ecuacion lineal del modelo
print("\nEcuacion del modelo\n")
print("Y= W0 + W1 . X \n")
print("Y="+str(lr.intercept_)+"+"+str(lr.coef_)+".x")
print("\nPresicion del modelo")
print(lr.score(x_train,y_train))
#Este algoritmo no se aplica eficientemente a estos tipos de datos
  
