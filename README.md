# Datasets Candidatos

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

* Nos asignan el problema de Facebook Comment Volume Dataset Data Set
* https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset

# Modelos de machine learning candidatos

* Linear Model
* Random Forest
* MLP

# Memoria desarrollada

* La memoria en la que analizamos los datos y resultados se encuentra en el siguiente [enlace](https://github.com/SergioQuijanoRey/PracticaFinalAprendizajeAutomatico/blob/master/Memoria/Memoria.pdf)

# Notas finales

* Por exigencias de los profesores, todo el código debía estar escrito en un único fichero `main.py`
    * De no haber sido así, habríamos escrito distintos módulos `.py` desarrollados a lo largo del curso, y un `jupyter notebook` donde realizar toda la exploración de datos, selección de modelo, entrenamiento y análisis de resultados
* Los modelos fueron entrenados en *Google Collab*. Para actualizar tanto los datos como el código de forma cómoda usamos `rclone` sobre el *Google Drive* proporcionado por la Universidad de Granada
