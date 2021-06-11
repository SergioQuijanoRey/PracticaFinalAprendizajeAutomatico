# Candidatos

1. Optical Recognition of Handwritten Digits Data Set -- 29
    * Vienen en pixeles, [0, 1]
    * Letter Recognition Data Set si nos quitan esta opción
    * Fácil de visualizar
    * Muchos ficheros
2. Facebook Comment Volume Dataset Data Set -- 28
    * 1000 filas de testing
    * 40.000 * 5 filas en training
    * Entendibles lo que son los atributos
3. Communities and Crime Data Set -- 2, 31
    * Hay que elegir entre muchos atributos
    * 1 archivo csv con 1994 filas
    * Bien comentado lo que es el dataset
4. Breast Cancer Wisconsin
    * Muy complicado
    * Esto para el chaval flipado, chico serio, si no buscas compromiso este no es tu chico. Limpio, ordenado.

# Asignación final

* Nos asignan el problema de Facebook Comment Volume Dataset Data Set -> Me quiero morir
* https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset

# TODO

* [x] Juntar los datasets
* [x] Limpieza de los datos
* [x] Pensar los modelos que queremos usar
* [x] Escribir la funcion que elimine outliers con el `z-score` de `Pandas`
* [x] Función que tome los resultados de `GridSearchCV` y los muestre por pantalla en un formato legible
* [x] Hacer los *cross-validation* faltantes
* [x] Adaptar Cross validation a no usar PCA
* [x] Adaptar Cross Validation + PCA a polinomios
* [ ] Entrenar sobre todo el conjunto de datos
* [ ] Escribir memoria 😥

# Dudas Mesejo

* [ ] Preguntar a Mesejo lo de las learning curves
* [ ] Se puede usar el out of bag error en RForest para acelerar el Cross Validation?

# Modelos candidatos

1. Modelo lineal: ajuste de un hiperplano
    * Lasso
    * Ridge
2. Perceptrón multicapa
3. Random Forrest
4. Radial Basis Function
5. Boosting con árboles simples se puede poner como ejemplo de que van mal, no están incorrelados, y esto es lo que trata de mejorar Random Forrest

# Notas

* Support vector machine, en los apuntes, ponía que no tenía mucho éxito con regresión
* Poner en la memoria cómo hemos hecho `unzip` en la carpeta de datos
* Hay que explicar lo que hace LocalOutlierFactor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

# Consejos Mesejo

* Cómo hacer que entrene bien:
    * Escoger bien las épocas, aplicar early stopping
    * Ser astutos a la hora de escoger parámetros
    * Con curvas de aprendizaje podemos saber cuándo parar los entrenamientos
    * Normalizar los datos acelera los entrenamientos
* Fijarnos en las curvas de entrenamiento: curvas de train y val a lo largo de las epochs
* A Nicolás le gustará que mostremos las curvas de aprendizaje
* Mas del 5% de los ejemplos fuera con outliers, está probablemente mal
* Podemos reducir a 70% - 30% de los datos para training - testing
