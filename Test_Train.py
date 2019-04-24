
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X ,y = iris.data , iris.target


#Creo mi clasificador
classifier = KNeighborsClassifier()

#Recommended division rates
# 2/3 Train 1/3 test
# 80% train 20% test 
# 50% / 50% 

train_X, test_X, train_y, test_y = train_test_split(
       
        X, y, train_size=0.5, test_size=0.5, random_state=123)

#Usar particiones estratificadas cuando los conjunto de datos sean pequeños para mantener la proporciones por defecto en el dataset por clase...
#mantenemos la proporción de datos por clase que había originalmente en los nuevos subconjuntos generados...

#Visualizo las proporciones
print('Todos:', np.bincount(y) )
print('Entrenamiento:', np.bincount(train_y) )
print('Test:', np.bincount(test_y) )

print("\nCon la propiedad stratify\n")

train_X, test_X, train_y, test_y = train_test_split(
       
        X, y, train_size=0.5, test_size=0.5, random_state=123, stratify=y)

print('Todos:', np.bincount(y) )
print('Entrenamiento:', np.bincount(train_y) )
print('Test:', np.bincount(test_y) )


#Dividir el dataset ayuda a que el modelo sea eficiente a la hora de clasificar, no memorizando, si no aprendiendo.

classifier.fit(train_X, train_y)

pred_y = classifier.predict(test_X)

print("\nExactitud de predicción del modelo: "+str(np.mean(pred_y == test_y)*100)+"%\n")

print("Ejemplos correctamente clasificados")
correct_idx = np.where(pred_y == test_y)[0]
print(correct_idx)

print("\nEjemplos incorrectamente clasificados")
incorrect_idx = np.where(pred_y != test_y)[0]
print(incorrect_idx)

#Representación

colors=["blue", "green", "gray"]

for n, color in enumerate(colors):
   
    idx = np.where(test_y==n)[0]
    plt.scatter(test_X[idx,1], test_X[idx,2],      color=color, label="Clase %s" % str(n))
    
plt.scatter(test_X[incorrect_idx, 1], test_X[incorrect_idx, 2], color="red", label="Malas predicciones")

print("\nRepresentación grafica")

plt.title("Resultados de clasificación en iris con KNN")
plt.xlabel("Ancho sépalo [Cm]")
plt.ylabel("Longitud pétalo [Cm]")
plt.legend(loc=5)
plt.show()

#Los errores se corresponden en las areas donde los verdes y grises se superponen. 
#Esto ayuda a guiarnos acerca de que caracteristicas habria que añadir para diferenciar mejor estas clases, ayudando a mejorar el rendimiento de nuestro modelo.
    
    
    















