import pandas as pd

# Parametros globales de core.py
# TODO -- poner a True
WAIT = False

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

    # TODO -- borrar este codigo comentado
    #stats["var"] = df.var()
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
    Añade un pandas.Series a un pandas.Dataframe
    Se usa como operacion inversa a split_dataset_into_X_and_Y, junta el dataframe de caracteristicas de
    prediccion y el dataframe de variables de salida en uno unico
    Parameters:
    ===========
    dataframe: dataframe con toda la matriz de datos
               Sera el dataframe con las caracteristicas de entrada
    series: pandas.Series con la columna que queremos añadir
            Sera la columna con la caracteristica a predecir
    column_name: el nombre de la columna que queremos añadir
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

    # Guardamos los resultados que nos interesan
    params = results["params"]
    mean_test_score = results["mean_test_score"]
    std_test_score = results["std_test_score"]

    # Mostramos los resultados
    for param, mean_score, std_score in zip(params, mean_test_score, std_test_score):
        print(f"\t--> Param: {param}, mean_score: {mean_score}, std_score: {std_score}")
    print("")

    wait_for_user_input()
