import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import fitter as ft
import pingouin as pg
import statsmodels.api as sm 
plt.style.use("ggplot")


def joiner_databases():

    files=["BEN_pro.csv", "CO_pro.csv", "MXIL_pro.csv", "NO_pro.csv", "NO2_pro.csv", "O3_pro.csv", "PM2.5_pro.csv", "SO2_pro.csv", "TOL_pro.csv"]

    datos = pd.DataFrame()

    for i in files:
        df = pd.read_csv(f"Database/{i}", delimiter=";", encoding="utf-8")
        
        var = df.iloc[:, 1]
        

        datos = pd.concat([datos, var], axis=1)
    
    file_names = [i.split("_")[0] for i in files]

    datos.columns = file_names

    datos.to_csv("datos.csv", sep=";", encoding="utf-8", index=False)
 
    return file_names


def correlation_matrix(file_names):

    files=["BEN_pro.csv", "CO_pro.csv", "MXIL_pro.csv", "NO_pro.csv", "NO2_pro.csv", "O3_pro.csv", "PM2.5_pro.csv", "SO2_pro.csv", "TOL_pro.csv"]

    datos = pd.DataFrame()

    for i in files:
        df = pd.read_csv(f"Database/{i}", delimiter=";", encoding="utf-8")
        
        var = df.iloc[:, 1]
        

        datos = pd.concat([datos, var], axis=1)

    corr = datos.corr()
    
    # if corr.empty:
    #     print("Está mal")
    # else:
    #     print(corr)

    fig, ax = plt.subplots(figsize=(9,9))

    fig.subplots_adjust(left=0.165)

    sns.heatmap(
        corr, 
        cmap='RdBu',
        annot=True,
        # annot_kws={"fontsize":10},
        # cbar=True
        vmin=-1, vmax=1, center=0,
        # cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        ax=ax,
    )

    xticks_position = [(i + 0.5) for i in range(len(file_names))]
    yticks_position = [(i + 0.5) for i in range(len(file_names))]
    

    # plt.xticks(range(len(file_names)), file_names, fontsize=10)
    # plt.yticks(range(len(file_names)), file_names, rotation = 0, fontsize=10)
    plt.xticks(xticks_position, file_names, fontsize=10)
    plt.yticks(yticks_position, file_names, rotation = 0, fontsize=10)
    plt.title("Matriz de correlación de variables")
    plt.show()


if __name__ == '__main__':
    file_names = joiner_databases()
    correlation_matrix(file_names)