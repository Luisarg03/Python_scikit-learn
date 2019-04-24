# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:53:00 2019

@author: Gorila
"""
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()

#Numero de ejemplos y numero de caracteristicas
samples, features = faces.data.shape
print((samples, features))

# Configurar la figura
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# mostrar algunas caras
for i in range(36):
     ax = fig.add_subplot(6,6, i + 1, xticks=[], yticks=[])
     ax.imshow(faces.images[i], cmap=plt.cm.bone)
     
