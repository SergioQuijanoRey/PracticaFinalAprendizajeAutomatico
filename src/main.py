"""
Authors:
    - Lucia Salamanca Lopez: TODO <- Pon tu correo
    - Sergio Quijano Rey: sergioquijano@correo.ugr.es
Dataset:
    - Facebook Comment Volume Dataset Data Set
    - https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carga de los datos
#===============================================================================
def load_data():
    """Cargamos todos los datos en un unico dataframe. Tanto de training como de test"""
    data_files = [
        "./datos/Training/Features_Variant_1.arff",
        "./datos/Training/Features_Variant_1.csv",
        "./datos/Training/Features_Variant_2.arff",
        "./datos/Training/Features_Variant_2.csv",
        "./datos/Training/Features_Variant_3.arff",
        "./datos/Training/Features_Variant_3.csv",
        "./datos/Training/Features_Variant_4.arff",
        "./datos/Training/Features_Variant_4.csv",
        "./datos/Training/Features_Variant_5.arff",
        "./datos/Training/Features_Variant_5.csv",
        "./datos/Testing/TestSet/Test_Case_1.arff",
        "./datos/Testing/TestSet/Test_Case_1.csv",
        "./datos/Testing/TestSet/Test_Case_2.arff",
        "./datos/Testing/TestSet/Test_Case_2.csv",
        "./datos/Testing/TestSet/Test_Case_3.arff",
        "./datos/Testing/TestSet/Test_Case_3.csv",
        "./datos/Testing/TestSet/Test_Case_4.arff",
        "./datos/Testing/TestSet/Test_Case_4.csv",
        "./datos/Testing/TestSet/Test_Case_5.arff",
        "./datos/Testing/TestSet/Test_Case_5.csv",
        "./datos/Testing/TestSet/Test_Case_6.arff",
        "./datos/Testing/TestSet/Test_Case_6.csv",
        "./datos/Testing/TestSet/Test_Case_7.arff",
        "./datos/Testing/TestSet/Test_Case_7.csv",
        "./datos/Testing/TestSet/Test_Case_8.arff",
        "./datos/Testing/TestSet/Test_Case_8.csv",
        "./datos/Testing/TestSet/Test_Case_9.arff",
        "./datos/Testing/TestSet/Test_Case_9.csv",
        "./datos/Testing/TestSet/Test_Case_10.arff",
        "./datos/Testing/TestSet/Test_Case_10.csv",
    ]

    df = pd.DataFrame()
    for data_file in data_files:
        current_df = pd.read_csv(data_file, header = None)
        df.append(current_df)
        break

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())

