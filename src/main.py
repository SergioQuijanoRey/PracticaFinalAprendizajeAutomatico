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
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from scipy import stats # Para calcular el z-score en un dataframe de pandas

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor

# ===================== FICHERO CORE =====================
import pandas as pd

# Parametros globales de core.py
WAIT = True

def calculate_stats(df):
    """
    Calcula un nuevo dataframe con las estadisticas relevantes del dataframe pasado como parametro
    Parameters:
    ===========
    df: dataframe del que queremos calcular algunas estadisticas

    Returns:
    ========
    stats: dataframe con las estadisticas calculadas
    """
    stats = pd.DataFrame()
    stats["type"] = df.dtypes
    stats["mean"] = df.mean()
    stats["median"] = df.median()

    stats["std"] = df.std()
    stats["min"] = df.min()
    stats["max"] = df.max()
    stats["p25"] = df.quantile(0.25)
    stats["p75"] = df.quantile(0.75)

    # Considero missing value algun valor que se null o que sea NaN (Not a Number)
    stats["missing vals"] = df.isnull().sum() + df.isna().sum()

    return stats

def print_full(df):
    """
    Muestra todos los datos de un pandas.DataFrame
    Codigo obtenido de
        https://stackoverflow.com/questions/19124601/pretty-print-an-entire-pandas-series-dataframe
    """
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def append_series_to_dataframe(dataframe, series):
    """
    A??ade un pandas.Series a un pandas.Dataframe
    Se usa como operacion inversa a split_dataset_into_X_and_Y, junta el dataframe de caracteristicas de
    prediccion y el dataframe de variables de salida en uno unico
    Parameters:
    ===========
    dataframe: dataframe con toda la matriz de datos
               Sera el dataframe con las caracteristicas de entrada
    series: pandas.Series con la columna que queremos a??adir
            Sera la columna con la caracteristica a predecir
    column_name: el nombre de la columna que queremos a??adir
    Returns:
    ========
    df: dataframe con los datos correctamente juntados
    """

    df = dataframe.copy()
    df.loc[:, "53"] = series.copy()
    return df

def print_bar(car = "=", width = 80):
    """Muestra por pantalla una barra horizontal"""
    print(car * width)

def wait_for_user_input():
    """Esperamos a que el usuario pulse una tecla para continuar"""
    if WAIT is True:
        input("Pulse una tecla para CONTINUAR...")

def split_dataset_into_X_and_Y(df):
    """
    Tenemos un dataframe con las variables dependientes y la variable dependiente. Esta funcion los
    separa en un dataframe para cada tipo de variable
    Parameters:
    ===========
    df: dataframe con los datos
    Returns:
    ========
    df_X: dataframe con las variables dependientes
    df_Y: dataframe con las variables a predecir (en este caso, una unica variable)
    """

    return df.loc[:, df.columns != "53"], df["53"]


def human_readable_results(results, title = ""):
    print(f"Resultados para {title}")
    print_bar()
    print("")
    print(results)

    # Guardamos los resultados que nos interesan
    params = results["params"]
    mean_test_score = results["mean_test_score"]
    mean_training_score = results["mean_train_score"]
    std_test_score = results["std_test_score"]

    # Mostramos los resultados
    for param, mean_score, std_score, mean_train_score in zip(params, mean_test_score, std_test_score, mean_training_score):
        print(f"\t--> Param: {param}, mean_score: {mean_score}, std_score: {std_score}, test_times_train = {mean_score / mean_train_score}")
    print("")

    wait_for_user_input()
# ========================================================

# Parametros globlales
#===============================================================================
n_jobs = -1 # Poner a None si peta demasiado
folds = 10  # Para controlar los tiempos de cross validation

# Carga de los datos
#===============================================================================
def load_data():
    """
    Cargamos todos los datos en un unico dataframe. Tanto de training como de test
        El codigo esta basado en:
        https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
    """
    data_files = [
        "./datos/Training/Features_Variant_1.csv",
        "./datos/Testing/TestSet/Test_Case_1.csv",

        # Estos path se usan cuando estamos ejecutando el codigo en Google Drive
        # Son los mismos archivos que los especificados arriba
        #"/content/drive/MyDrive/ml/datos/Training/Features_Variant_1.csv",
        #"/content/drive/MyDrive/ml/datos/Testing/TestSet/Test_Case_1.csv",
    ]

    dfs = (pd.read_csv(data_file, header = None) for data_file in data_files)
    df = pd.concat(dfs, ignore_index=True)
    return df

def split_train_test(df):
    """Dado un dataframe con todos los datos de los que disponemos, separamos en training y test"""
    # Dividimos en los datos y los valores
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Dividimos en training y test (80% y 20%)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle=True, random_state = 123456789)
    return x_train, x_test, y_train, y_test

# Exploracion de los datos
#===============================================================================
def explore_training_set(df, show_plot = False):
    """
    Muestra caracteristicas relevantes del dataset de entrenamiento

    Parameters:
    ===========
    df: dataframe del que queremos realizar la exploracion
        No debe contener datos de test, pues no queremos visualizarlos
    show_plot: Indica si queremos que se muestren graficos o no

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

    print("Boxplot de la variable de salida")
    plot_boxplot(df, title = "Boxplot de la variable de salida", columns = [df.columns[-1]])
    wait_for_user_input()

def plot_boxplot(df, columns, title):
    """
    Hacemos graficos de barras de un dataframe
    Parameters:
    ===========
    df: el dataframe que contiene los datos
    columns: lista con los nombres de las columnas de las que queremos hacer la grafica
    title: titulo de la grafica
    """

    boxplot = df.boxplot(column=columns)
    plt.title(title)
    plt.show()
    wait_for_user_input()


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
    train_df: datos de entrenamiento, de los que se calculan los estadisticos para la
              transformacion
    test_df: datos de test. No se toma informacion de esta muestra para calcular la
             trasnformacion
    Returns:
    ========
    standarized_train: datos de entrenamiento estandarizados
    standarized_test: datos de test estandarizados con la misma transformacion
                     calculada a partir de los datos de entrenamiento
    """

    scaler = StandardScaler()
    standarized_train = scaler.fit_transform(train_df)
    standarized_test = scaler.transform(test_df)

    return standarized_train, standarized_test

def remove_outliers(df_train_x, df_train_y):
    """Elimina los outliers del dataset
    """

    df_train_x = df_train_x.to_numpy()
    df_train_y = df_train_y.to_numpy()

    # Modelo que usamos para detectar outliers
    cov = EllipticEnvelope(contamination = 0.05).fit(df_train_x)

    # Indices que nos dicen si tenemos outliers o no
    outliers_indixes = cov.predict(df_train_x)

    # Borramos segun estos indices
    mask = outliers_indixes == 1
    df_train_x = df_train_x[mask, :]
    df_train_y = df_train_y[mask]

    return df_train_x, df_train_y


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

def show_cross_validation(df_train_x, df_train_y, df_train_x_original):
    """
    Lanza cross validation y muestra los resultados obtenidos

    Parameters:
    ===========
    df_train_X: dataframe con los datos de entrada, a los que hemos aplicado PCA y polinomios
    df_train_Y: dataframe con los datos de salida
    df_train_X_original: dataframe con los datos sin aplicar PCA
    """

    # Por si acaso no hemos hecho previamente la conversion
    try:
        df_train_x = df_train_x.to_numpy()
        df_train_y = df_train_y.to_numpy()
    except:
        pass

    print("--> CV -- PCA + Polinimio orden 2")
    # Cross validation para modelos lineales
    cross_validation_linear(df_train_x, df_train_y)

    # Cross validation para SVM
    cross_validation_mlp(df_train_x, df_train_y)

    # Cross validation para random forest
    cross_validation_random_forest(df_train_x, df_train_y)

    print("--> CV -- No PCA")
    # Cross validation para modelos lineales
    cross_validation_linear(df_train_x_original, df_train_y)

    # Cross validation para MLP
    cross_validation_mlp(df_train_x_original, df_train_y)

    # Cross validation para random forest
    cross_validation_random_forest(df_train_x_original, df_train_y)

    wait_for_user_input()

def cross_validation_linear(df_train_X, df_train_Y):
    # Parametros prefijados
    max_iters = 1e4
    tol = 1e-4

    # Kfold cross validation
    kf = KFold(n_splits=folds, shuffle = True)

    # Los dos modelos lineales que vamos a considerar
    lasso = Lasso(max_iter = max_iters, tol = tol)
    ridge = Ridge(max_iter = max_iters, tol = tol, solver = "cholesky")

    # Espacio de busqueda
    parameters = {
        'alpha': [10**x for x in [-1, 0]] + np.linspace(1, 2, 2).tolist()
    }

    # CV para Lasso
    gs = GridSearchCV(lasso, parameters, scoring = "neg_mean_squared_error", cv = kf, refit = False, verbose = 3, n_jobs = n_jobs, return_train_score = True)
    gs.fit(df_train_X, df_train_Y)
    results = gs.cv_results_
    human_readable_results(results, title = "Lasso")

    # CV para Ridge
    gs = GridSearchCV(ridge, parameters, scoring = "neg_mean_squared_error", cv = kf, refit = False, verbose = 3, n_jobs = n_jobs, return_train_score = True)
    gs.fit(df_train_X, df_train_Y)
    results = gs.cv_results_
    human_readable_results(results, title = "Ridge")

def cross_validation_random_forest(df_train_X, df_train_Y):
    # Kfold cross validation
    kf = KFold(n_splits=folds, shuffle = True)

    # Modelo que vamos a considerar
    # n_jobs para indicar que queremos usar mas de un procesador en el entrenamiento
    randomForest = RandomForestRegressor(criterion="mse", bootstrap=True, max_features = "sqrt", n_jobs = n_jobs)

    # Espacio de busqueda
    parameters = {
        # Numero de arboles
        'n_estimators': np.array([90, 95, 100, 120]),
        'max_depth': np.array([5, 10]),
        'min_samples_leaf': np.array([2, 3])
    }

    gs = GridSearchCV(randomForest, parameters, scoring = "neg_mean_squared_error", cv = kf, refit = False, verbose = 3, n_jobs = n_jobs, return_train_score = True)
    gs.fit(df_train_X, df_train_Y)
    results = gs.cv_results_
    human_readable_results(results, title="Random Forest")

def cross_validation_mlp(df_train_X, df_train_Y):
    # Parametros prefijados
    layer_sizes = [(50, ), (75, ), (100,)]
    tol = 1e-4

    # Kfold cross validation
    kf = KFold(n_splits=folds, shuffle = True)

    # Modelo que vamos a considerar
    mlp = MLPRegressor(tol = tol, solver="adam", learning_rate_init = 0.001, early_stopping = True)

    # Espacio de busqueda
    parameters = {
        'alpha': [10**x for x in [-2, -1, 0]],
        'activation': ['relu'],
        'hidden_layer_sizes': layer_sizes
    }

    gs = GridSearchCV(mlp, parameters, scoring = "neg_mean_squared_error", cv = kf, refit = False, verbose = 3, n_jobs = n_jobs, return_train_score = True)
    gs.fit(df_train_X, df_train_Y)
    results = gs.cv_results_
    human_readable_results(results, title="MLP")

def show_results(model, df_train_x, df_train_y, df_test_x, df_test_y):
    """Mostramos los resultados del entrenamiento de un modelo"""
    # Modelo Dummy con constante 0 para usarlo como baseline
    dummy = DummyRegressor(strategy="constant", constant=0)
    dummy.fit(df_train_x, df_train_y)

    # Modelo Dummy con constante la media para usarlo como baseline
    dummy_mean = DummyRegressor(strategy="mean")
    dummy_mean.fit(df_train_x, df_train_y)

    # Realizamos las predicciones con el modelo
    train_predictions = model.predict(df_train_x)
    test_predictions = model.predict(df_test_x)
    dummy_predictions = dummy.predict(df_test_x)
    dummy_mean_predictions = dummy_mean.predict(df_test_x)

    train_r2 = r2_score(df_train_y, train_predictions)
    train_mse = mean_squared_error(df_train_y, train_predictions)
    train_mae = mean_absolute_error(df_train_y, train_predictions)

    test_r2 = r2_score(df_test_y, test_predictions)
    test_mse = mean_squared_error(df_test_y, test_predictions)
    test_mae = mean_absolute_error(df_test_y, test_predictions)

    dummy_r2 = r2_score(df_test_y, dummy_predictions)
    dummy_mse = mean_squared_error(df_test_y, dummy_predictions)
    dummy_mae = mean_absolute_error(df_test_y, dummy_predictions)

    dummy_mean_r2 = r2_score(df_test_y, dummy_mean_predictions)
    dummy_mean_mse = mean_squared_error(df_test_y, dummy_mean_predictions)
    dummy_mean_mae = mean_absolute_error(df_test_y, dummy_mean_predictions)

    # Mostramos los resultados
    print("--> Resultados en training set:")
    print(f"\t--> MSE: {train_mse}")
    print(f"\t--> MAE: {train_mae}")
    print(f"\t--> R2: {train_r2}")
    print("")
    print("--> Resultados en testing set:")
    print(f"\t--> MSE: {test_mse}")
    print(f"\t--> MAE: {test_mae}")
    print(f"\t--> R2: {test_r2}")
    print("")
    print("--> Resultados en dummy constante 0:")
    print(f"\t--> MSE: {dummy_mse}")
    print(f"\t--> MAE: {dummy_mae}")
    print(f"\t--> R2: {dummy_r2}")
    print("")
    print("--> Resultados en dummy constante media:")
    print(f"\t--> MSE: {dummy_mean_mse}")
    print(f"\t--> MAE: {dummy_mean_mae}")
    print(f"\t--> R2: {dummy_mean_r2}")
    print("")
    wait_for_user_input()

def learning_curve(model, df_train_x, df_train_y, df_test_x, df_test_y, number_of_splits = 10):
    """
    Codigo inspirado de: https://github.com/rasbt/mlxtend/blob/master/mlxtend/plotting/learning_curves.py
    """

    training_errors = []
    testing_errors = []

    ranges = [int(i) for i in np.linspace(0, df_train_x.shape[0], number_of_splits)][1:]
    for i, range in enumerate(ranges):
        print(f"--> Empezando el paso {i}")
        # Datos con los que trabajamos en este paso
        current_train_x = df_train_x[:range]
        current_train_y = df_train_y[:range]

        # Entrenamos en este paso
        model.fit(current_train_x, current_train_y)

        # Calculamos los errores
        curr_training_error = mean_squared_error(current_train_y, model.predict(current_train_x))
        curr_test_error = mean_squared_error(df_test_y, model.predict(df_test_x))
        training_errors.append(curr_training_error)
        testing_errors.append(curr_test_error)

    plt.title("Curva de aprendizaje")
    plt.plot(ranges, training_errors, "tab:blue", label = "Error de entrenamiento")
    plt.plot(ranges, testing_errors, "tab:orange", label = "Error de testing")
    plt.legend(loc='upper right',shadow=True)
    plt.xlabel("Cantidad de datos en los que se entrena")
    plt.ylabel(f"Error Cuadratico medio")
    plt.show()
    wait_for_user_input()

def where_did_the_model_failed(model, df_x, df_y):
    """Mostramos los puntos en los que mas ha fallado el modelo"""

    # Realizamos esta conversion por seguridad
    try:
        df_x = df_x.to_numpy()
    except:
        pass
    try:
        df_y = df_y.to_numpy()
    except:
        pass

    # Realizamos las predicciones
    predictions = model.predict(df_x)

    # Calculamos el error cometido
    diff = df_y - predictions
    error_vector = diff ** 2

    # Ordenamos los errores obtenidos de mayor a menor
    indixes = np.argsort(error_vector)

    # Nos quedamos con los primeros 100 elementos donde fallamos mas
    worst_indixes = [indixes[-i] for i in range(1, 25 + 1)]

    worst_elements = df_y[np.array(worst_indixes)]
    worst_predictions = predictions[np.array(worst_indixes)]
    print("Los peores elementos son:")
    print(worst_elements)
    wait_for_user_input()

    plt.title("Peores valores de salida predichos")
    plt.scatter(worst_elements, worst_predictions)
    plt.xlabel("Peores etiquetas reales")
    plt.ylabel("Peores etiquetas predichas")
    plt.show()
    wait_for_user_input()


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
    print(f"--> N?? datos entrenamiento: {len(df_train_x)}")
    print(f"--> N?? datos test: {len(df_test_x)}")


    # Exploramos el conjunto de entrenamiento
    print("==> Exploramos el conjunto de entrenamiento")
    explore_training_set(append_series_to_dataframe(df_train_x, df_train_y), show_plot = True)

    # Borramos los outliers
    print("==> Borrando outliers")

    # Para saber cuantas filas estamos borrando
    prev_len = len(df_train_x)

    # Eliminamos los outliers
    df_train_x, df_train_y = remove_outliers(df_train_x, df_train_y)

    print(f"Tama??o tras la limpieza de outliers del train_set: {len(df_train_x)}")
    print(f"Shapes de X e Y: {df_train_x.shape}, {df_train_y.shape}")
    print(f"Numero de filas eliminadas: {prev_len - len(df_train_x)}")
    print(f"Porcentaje de filas eliminadas: {float(prev_len - len(df_train_x)) / float(prev_len) * 100.0}%")
    wait_for_user_input()

    print("==> Estandarizando el dataset")
    df_train_x, df_test_x = standarize_dataset(df_train_x, df_test_x)
    print("--> Estadisticas tras la estandarizacion")
    explore_training_set(pd.DataFrame(df_train_x))

    print("==> Aplicando PCA")

    # Guardamos estos dataframes para futuras comparaciones
    df_train_original_x = df_train_x.copy()
    df_test_original_x = df_test_x.copy()

    df_train_x, df_test_x = apply_PCA(df_train_x, df_test_x, explained_variation = 0.99)

    print("==> Aplicando polinomio grado 2 al conjunto PCA")
    poly = PolynomialFeatures(2)
    df_train_x = poly.fit_transform(df_train_x)
    df_test_x = poly.transform(df_test_x)

    # Tenemos que estandarizar el conjunto PCA + Pol para que el entrenamiento no tarde tanto
    df_train_x, df_test_x = standarize_dataset(df_train_x, df_test_x)

    print("==> Lanzando cross validation")
    # Esta comentado porque tarda demasiado tiempo en ejecutarse
    #  show_cross_validation(df_train_x, df_train_y, df_train_original_x)

    print("==> Entrenando sobre todo el conjunto de datos sin PCA")
    model = RandomForestRegressor(criterion="mse", bootstrap=True, max_features = "sqrt", max_depth = 10, min_samples_leaf = 2, n_estimators = 90, n_jobs = n_jobs)
    print(f"--> Entrenando sobre todo el conjunto de datos con el modelo final")
    model.fit(df_train_original_x, df_train_y)

    print(f"--> Modelo entrenado, mostrando resultados")
    show_results(model, df_train_original_x, df_train_y, df_test_original_x, df_test_y)

    print(f"--> Mostrando la curva de aprendizaje del entrenamiento del modelo")
    learning_curve(model, df_train_original_x, df_train_y, df_test_original_x, df_test_y, number_of_splits = 10)

    print(f"--> Mostrando donde ha fallado nuestro modelo")
    where_did_the_model_failed(model, df_test_original_x ,df_test_y)

    print("==> Entrenando sobre todo el conjunto de datos con el baseline MLP")
    baseline = MLPRegressor(alpha = 1, hidden_layer_sizes = [(100)], activation = "relu", tol = 1e-4, solver="adam", learning_rate_init = 0.001, early_stopping = True)
    baseline.fit(df_train_original_x, df_train_y)

    print(f"--> Modelo entrenado, mostrando resultados")
    show_results(baseline, df_train_original_x, df_train_y, df_test_original_x, df_test_y)

    print(f"--> Mostrando la curva de aprendizaje del entrenamiento del baseline")
    learning_curve(baseline, df_train_original_x, df_train_y, df_test_original_x, df_test_y, number_of_splits = 10)
