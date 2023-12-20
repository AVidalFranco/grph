import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')

# Definimos as funcións
def labeler(varname):
     if varname == "BEN":
         label_title = "BENCENO"
     elif varname == "CO":
        label_title = "CO"
     elif varname == "MXIL":
        label_title  = "M-XILENO"
     elif varname == "NO2":
        label_title = "NO2"
     elif varname == "O3":
        label_title = "O3"
     elif varname == "PM2.5":
         label_title = "PM2.5"
     elif varname == "SO2":
         label_title = "SO2"
     elif varname == "TOL":
         label_title = "TOLUENO"
     return label_title

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

def violins(file_names):
    datos = pd.read_csv("datos.csv", delimiter=";", encoding="utf-8")
   
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    datos = scaler.fit_transform(datos)
    print(datos)

    #Gráficas de violín

    fig, ax = plt.subplots(figsize=(4, 4))
    # fig.subplots_adjust(bottom)
    sns.set(style="whitegrid")
    sns.violinplot(data=datos)
    # label_title = labeler(varname=file_name)
    # plt.xlabel(file_names)
    # plt.ylabel("Valores")
    # plt.xticks(rotation=45)
    # plt.title("Gráficas de violín")
    plt.xticks(range(len(file_names)), file_names)
    plt.xlabel("Variables")
    plt.ylabel("Valores")
    plt.title("Gráficas de violín")
    plt.show()



        
if __name__ == '__main__':
    file_names = joiner_databases()
    violins(file_names)
    

