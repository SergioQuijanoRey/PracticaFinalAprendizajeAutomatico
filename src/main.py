"""
Authors:
    - Lucia Salamanca Lopez: luciasalamanca@correo.ugr.es
    - Sergio Quijano Rey: sergioquijano@correo.ugr.es
Dataset:
    - Facebook Comment Volume Dataset Data Set
    - https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats # Para calcular el z-score en un dataframe de pandas
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

# TODO -- borrar esto y copiar el archivo core aqui
from core import *

# Carga de los datos
#===============================================================================
def load_data():
    """Cargamos todos los datos en un unico dataframe. Tanto de training como de test"""
    data_files = [
        "./datos/Training/Features_Variant_2.csv",
        "./datos/Training/Features_Variant_3.csv",
        "./datos/Training/Features_Variant_4.csv",
        "./datos/Training/Features_Variant_5.csv",
        "./datos/Testing/TestSet/Test_Case_1.csv",
        "./datos/Testing/TestSet/Test_Case_2.csv",
        "./datos/Testing/TestSet/Test_Case_3.csv",
        "./datos/Testing/TestSet/Test_Case_4.csv",
        "./datos/Testing/TestSet/Test_Case_5.csv",
        "./datos/Testing/TestSet/Test_Case_6.csv",
        "./datos/Testing/TestSet/Test_Case_7.csv",
        "./datos/Testing/TestSet/Test_Case_8.csv",
        "./datos/Testing/TestSet/Test_Case_9.csv",
        "./datos/Testing/TestSet/Test_Case_10.csv",
    ]

    df = pd.read_csv("./datos/Training/Features_Variant_1.csv", header = None)
    for data_file in data_files:
        current_df = pd.read_csv(data_file, header = None)
        df.append(current_df)
    
    return df

def split_train_test(df):
    """Dado un dataframe con todos los datos de los que disponemos, separamos en training y test"""
    # Dividimos en los datos y los valores
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Dividimos en training y test (80% y 20%)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle=True)
    return x_train, x_test, y_train, y_test

# Exploracion de los datos
#===============================================================================
def explore_training_set(df):
    """
    Muestra caracteristicas relevantes del dataset de entrenamiento
    Parameters:
    ===========
    df: dataframe del que queremos realizar la exploracion
        No debe contener datos de test, pues no queremos visualizarlos
    Returns:
    ========
    stats: dataframe con las estadisticas calculadas
    """

    # Extreamos algunas estadisticas de este dataframe
    stats = calculate_stats(df)

    print("Estadisticas del dataset de entrenamiento:")
    print_bar()
    print_full(stats)
    wait_for_user_input()

# Limpieza de datos
def remove_outliers(df, times_std_dev, output_cols = []):
    """
    Elimina las filas de la matriz representada por df en las que, en alguna columna, el valor de la
    fila esta mas de times_std_dev veces desviado de la media
    Paramters:
    ==========
    df: dataframe sobre el que trabajamos
    times_std_dev: umbral de desviacion respecto a la desviacion estandar
                   Un valor usual es 3.0, porque el 99.74% de los datos de una distribucion normal
                   se encuentran en el intervalo (-3 std + mean, 3 std + mean)
    output_cols: columnas de salida, sobre las que no queremos comprobar los outliers
    Returns:
    ========
    cleaned_df: dataframe al que hemos quitado las filas asociadas a los outliers descritos
    """

def outliers_out(df_x_train, df_y_train):
    """
    Borramos los outliers de nuestro dataset. Usando la técnica local factors

    Parameters:
    ===========
    df_x_train: dataframe con los datos de entrada
    df_y_train: dataframe con los valores de salida

    Returns:
    =========
    x_cleaned: dataframe con los datos de entrada sin outliers
    y_cleaned: dataframe con los datos de salida en los que hemos eliminado las mismas filas que en x_cleaned
    """
    # Identificamos los outliers en los datos
    # Pasamos a array los datasets
    np_x_train = df_x_train.to_numpy()
    np_y_train = df_y_train.to_numpy()

    lof = LocalOutlierFactor()
    outliers = lof.fit_predict(np_x_train)

    mask = outliers != -1

    # seleccionamos las filas que no son outliers
    x_cleaned = pd.DataFrame(np_x_train[mask, :])
    y_cleaned =  pd.Series(np_y_train[mask])
    return x_cleaned, y_cleaned

def remove_outliers(df, times_std_dev, output_cols = []):
    """
    Elimina las filas de la matriz representada por df en las que, en alguna columna, el valor de la
    fila esta mas de times_std_dev veces desviado de la media.

    Parameters:
    ==========
    df: dataframe sobre el que trabajamos
    times_std_dev: umbral de desviacion respecto a la desviacion estandar
                   Un valor usual es 3.0, porque el 99.74% de los datos de una distribucion normal
                   se encuentran en el intervalo (-3 std + mean, 3 std + mean)
    output_cols: columnas de salida, sobre las que no queremos comprobar los outliers
    Returns:
    ========
    cleaned_df: dataframe al que hemos quitado las filas asociadas a los outliers descritos
    """
    # Quitamos las columnas de salida al dataframe. Se usa para la siguiente linea en la que hacemos
    # la seleccion
    df_not_output = df

    # Filtramos las columnas, columna por columna
    for col in output_cols:
        df_not_output = df_not_output.loc[:, df_not_output.columns != col]

    # Filtramos los outliers, sin tener en cuenta las columnas de variables de salida
    return df[(np.abs(stats.zscore(df_not_output)) < times_std_dev).all(axis=1)]

def standarize_dataset(train_df, test_df):
    """
    Estandariza el dataset, usando solo la informacion de los datos de entrenamiento. A los datos de
    test se les aplica la misma transformacion. Notar que no se esta usando informacion de la
    muestra de test para aplicar la estandarizacion. Pero pasamos el conjunto de test para aplicar
    la misma trasnformacion a estos datos
    Si no queremos standarizar las columnas de salida, separar antes los dataframes y pasar solo
    la matriz de datos de entrada!!
    Parameters:
    ===========
    train_df: dataframe de datos de entrenamiento, de los que se calculan los estadisticos para la
              transformacion
    test_df: dataframe de datos de test. No se toma informacion de esta muestra para calcular la
             trasnformacion
    Returns:
    ========
    standarized_train: dataframe con los datos de entrenamiento estandarizados
    standarized_test: dataframe con los datos de test estandarizados con la misma transformacion
                     calculada a partir de los datos de entrenamiento
    """
    # Guardamos los nombres de las columna del dataframe, porque la tranformacion va a hacer que
    # perdamos este metadato
    prev_cols = train_df.columns

    scaler = StandardScaler()
    standarized_train = scaler.fit_transform(train_df)
    standarized_test = scaler.transform(test_df)

    # La transformacion devuelve np.arrays, asi que volvemos a dataframes
    standarized_train = pd.DataFrame(standarized_train, columns = prev_cols)
    standarized_test = pd.DataFrame(standarized_test, columns = prev_cols)

    return standarized_train, standarized_test

def apply_PCA(df_train_X, df_test_X, explained_variation = 0.90, number_components = None):
    """
    Aplica PCA al conjunto de entrada de los datos de entrenamiento

    Parameters:
    ===========
    df_train_X: dataframe con los datos de entrada de entrenamiento
                Importante: no debe contener la variable de salida
    df_test_X: dataframe con los datos de entrada de test. Solo los queremos para aplicar la misma
               transformacion que a los datos de entrada. No los usamos en el proceso de calcular
               la trasnformacion
    explained_variation: varianza explicada por los datos transformados que queremos alcanzar
                         Se aplica solo cuando number_components == None
    number_components: numero de componentes que queremos obtener, independientemente de la varianza
                       explicada obtenida. Es opcional
    Returns:
    ========
    df_transformed_X: datos de entrenamiento transformados
    df_test_transformed_X: datos de test transformados usando la misma transformacion calculada a
                           partir de los datos de entrenamiento
    """

    # Comprobacion de seguridad
    if type(explained_variation) is not float:
        raise Exception("El porcentaje de variabilidad explicada debe ser un flotante")

    # Si tenemos numero de componentes, no hacemos caso a la varianza explicada
    pca = None
    if number_components is not None:
        pca = PCA(number_components)
    else:
        # Queremos que PCA saque tantas dimensiones como porcentaje de variacion explidada especificado
        pca = PCA(explained_variation)

    # Nos ajustamos a la muestra de datos de entrenamiento
    print("Ajustando los datos de entrenamiento a la transformacion")
    pca.fit(df_train_X)

    # Aplicamos la transformacion al conjunto de entrenamiento y de test
    # Usamos variables para que no se modifiquen los dataframes pasados como parametro
    df_transformed_X = pca.transform(df_train_X)
    df_test_transformed_X = pca.transform(df_test_X)

    # Recuperamos los datos en formato dataframe
    # No podemos ponerle nombres a las columnas porque PCA mezcla las columnas sin tener nosotros
    # control sobre como se hace la transformacion
    df_transformed_X = pd.DataFrame(df_transformed_X)
    df_test_transformed_X = pd.DataFrame(df_test_transformed_X)

    # Mostramos algunos datos por pantalla
    print(f"Ajuste realizado:")
    print(f"\tPorcentaje de la varianza explicado: {pca.explained_variance_ratio_}")
    print(f"\tPorcentaje de la varianza explicado total: {sum(pca.explained_variance_ratio_)}")
    print(f"\tNumero de dimensiones obtenidas: {len(df_transformed_X.columns)}")
    wait_for_user_input()

    return df_transformed_X, df_test_transformed_X



# Funcion principal
#===============================================================================
if __name__ == "__main__":
    # Establecemos la semilla inicial
    print("==> Estableciendo semilla inicial")
    np.random.seed(123456789)

    # Cargamos los datos. Cargamos en un solo dataframe los datos de testing y de training
    print("==> Cargamos todos los archivos en un unico dataframe")
    df = load_data()

    # Separamos en training y test
    print("==> Separando en train y test")
    df_train_x, df_test_x, df_train_y, df_test_y = split_train_test(df)

    # Exploramos el conjunto de entrenamiento
    print("==> Exploramos el conjunto de entrenamiento")
    explore_training_set(append_series_to_dataframe(df_train_x, df_train_y, column_name=["53"]))

    # Borramos los outliers
    print("==> Borrando outliers")

    # Para saber cuantas filas estamos borrando
    prev_len = len(df_train_x)

    # TODO -- conseguir que esto funcione mas o menos bien
    #df = remove_outliers(append_series_to_dataframe(df_train_x, df_train_y, column_name=["53"]), 4.0, output_cols=["53"])
    #df_train_x, df_train_y = split_dataset_into_X_and_Y(df)

    df_train_x, df_train_y = outliers_out(df_train_x, df_train_y)
    print(f"Tamaño tras la limpieza de outliers del train_set: {len(df_train_x)}")
    print(f"Numero de filas eliminadas: {prev_len - len(df_train_x)}")
    print(f"Porcentaje de filas eliminadas: {float(prev_len - len(df_train_x)) / float(prev_len) * 100.0}%")
    wait_for_user_input()

    print("==> Estandarizando el dataset")
    df_train_x, df_test_x = standarize_dataset(df_train_x, df_test_x)
    # TODO -- mostrar en la memoria como queda estandarizado -> Por que? 
    # Porque estandarizando tambien se normalizan los rangos en cierta medida y eso
    # hay que justificarlo
    # TODO -- descomentar cuando hagamos la memoria
    #explore_training_set(df_train_x)

    print("==> Aplicando PCA")
    df_train_x, df_test_x = apply_PCA(df_train_x, df_test_x, explained_variation = 0.99)