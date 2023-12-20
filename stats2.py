import os
import numpy as np
import pandas as pd
import fitter as ft
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm 
import math


from scipy import stats
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import lognorm
from matplotlib import pyplot as plt
plt.style.use('ggplot')

"""This file contains functions for statistical analysis"""

def basic_stats(data):
    # Data has to be a 1D item: list, array, pd.DataFrame with one column or pd.Series...
    
    Maxim = max(data) # Max value of the variable
    Minim = min(data) # Min value of the variable
    Mean = np.mean(data) # Mean of the data studied
    Mode = stats.mode(data) # Mode of the data studied
    Std = np.std(data) # Standard deviation of the data
    Variance = np.var(data) # Variance of the data

    Q1 = np.percentile(data, 25) # Primer quartile
    Q2 = np.percentile(data, 50) # Segundo quartile
    Q3 = np.percentile(data, 75) # Tercer quartile
    InQR = Q3 - Q1 # Inter quartile range
    
    return Maxim, Minim, Mean, Mode, Std, Variance, Q1, Q2, Q3, InQR

def limit_checker():
    """This function implements the detection of those value
    above the legal limits of pollution in water bodies."""

    File = 'data_pro.csv'
    df = pd.read_csv(f'Database/{File}', delimiter=';', parse_dates=['date'], index_col=['date'])

    # Get those rows above the limit
    result = df.loc[df['ammonium'] >= 0.2]

    # In the case a multivariate condition
    result = df.loc[(df['ammonium'] >= 0.2) & (df['conductivity'] > 875)]

    print(result)

def nan_checker():
    """This funciton is used to check if there are NaNs in the db"""

    # Read the database
    File = 'Oxigeno disuelto'

    fileName, fileExtension = os.path.splitext(File)
    df = pd.read_csv(f'Database/{fileName}_pro.csv', delimiter=';', parse_dates=['date'], index_col=['date'])

    # Check for NaN in a single df column
    print('Are there any NaNs in the column?', df['value'].isnull().values.any())

    # Count the NaN under a single data frame colum
    print('Total number of NaNs in the df', df['value'].isnull().sum())

    # Check for NaN under an entire data frame
    print('Are there any NaNs in the df?', df.isnull().values.any())

    # Count the NaN under an entire data frame
    print('Total number of NaN in the df ', df.isnull().sum().sum())

    print('Index of the NaNs', df.index[df.isnull().any(1)])

    # Get the index where there is an inf
    print('Index where there is an inf', df.index[np.isinf(df).any(1)])

def number_bins(data):
    # Gets the number of bins for a specific variable (data)
    # Data has to be a 1D item: list, array, pd.DataFrame with one column or pd.Series...
    
    data = data.to_list()
    n = len(data) # number of observations
    range = max(data) - min(data) 
    numIntervals = np.sqrt(n) 
    width = range/numIntervals # width of the intervals
    
    return np.arange(min(data), max(data), width).tolist()

def correlation_matrix():
    """This function gets the correlation matrix. It would need to take a database with the 
    weeks in common in all variables after processing. Something like joiner.py"""
    # Load the data into a dataframe
    File = 'data_pro.csv'
    df = pd.read_csv(f'Database/{File}', delimiter=';', parse_dates=['date']) # Got rid off the date column as index because it is needless

    # Select the desired coluns
    cols = ['ammonium', 'conductivity', 'nitrates', 'oxygen', 'pH', 'temperature', 'turbidity', 'flow', 'pluviometry']

    df = df[cols]

    corr_matrix = df.corr().round(4)
    print(corr_matrix)

    sns.heatmap(corr_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
    plt.show()

def boxplot(data, Q1, Q3, InQR):
    
    plt.boxplot(data)
    
    upperBound = Q3 + (1.5*InQR)
    lowerBound = Q1 - (1.5*InQR)
    
    outliersBoxplot = ((data <= lowerBound) | (data >= upperBound))
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    print('These are the outliers in the boxplot:', outliersBoxplot)
    print(len(outliersBoxplot))
    plt.show()
    
    return outliersBoxplot

def plot_histogram(data):
    n, bins, patches = plt.hist(x = data, bins = 'auto', color = '#0504aa', alpha = 0.7, rwidth = 0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Combined probability histogram')
    maxfreq = n.max()

    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    plt.show()

# Statistical tests
def kolmogorov_smirnov_test(data):

    # Define the names of the distributions to analyze
    # distributions = ['alpha', 'anglit', 'arcsine', 'argus', 
    #     'bradford', 'burr', 'burr12',
    #     'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 
    #     'dgamma', 'dweibull', 
    #     'expon', 'exponnorm', 'exponweib', 'exponpow',
    #     'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm',
    #     'genlogistic', 'genpareto', 'gennorm', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'geninvgauss', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l',
    #     'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant',
    #     'invgamma', 'invgauss', 'invweibull',
    #     'johnsonsb', 'johnsonsu',
    #     'kappa3', 'kappa4', 'kstwobign',
    #     'laplace', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'loguniform', 'lomax',
    #     'maxwell', 'mielke', 'moyal',
    #     'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'norminvgauss',
    #     'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm',
    #     'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss',
    #     'semicircular', 
    #     't', 'trapezoid', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 
    #     'uniform', 
    #     'vonmises', 'vonmises_line', 
    #     'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']
    
    distributions = ['alpha', 'arcsine', 'beta', 'betaprime', 'burr', 'cauchy', 'expon', 'f', 'fisk', 'foldcauchy', 'gamma', 'genexpon', 'genextreme', 'geninvgauss', 'genpareto', 'gilbrat', 'gumbel_r', 'halfcauchy', 'halfgennorm', 'halfnorm', 'invgamma', 'invgauss', 'invweibull', 'kappa3', 'ksone', 'laplace', 'lognorm', 'loguniform', 'lomax', 'ncf', 'norm', 'norminvgauss', 'pareto', 'powerlognorm', 'rayleigh', 'recipinvgauss', 'reciprocal', 'skewcauchy', 'truncexpon', 'wald']
    
    results = {}
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        
        # Get the parameters of each distribution
        param = dist.fit(data)

        # Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        
        # Store the resuts of the test and the parameters
        results[dist_name] = p, param
        # print('Done', dist_name)

    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    counter = 1
    for i in list(results.items())[0:5]:
        print(f'Top {counter} distribution:', i[0], 'with p-value:', i[1][0])
        
        counter += 1

def andserson_darling_test(data):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
    # Define the names of the distributions to analyze
    # distributions = ['norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1']
    distributions = ['norm', 'expon', 'logistic', 'gumbel', 'gumbel_r']
    
    results = {}
    for distribution in distributions:
        s = stats.anderson(data, dist=distribution)

        results[distribution] = s
    
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    counter = 1
    for i in list(results.items())[0:5]:
        print(f'Top {counter} distribution:', i[0], 'with p-value:', i[1][0])

        counter += 1

def fitter_test(data):

    # distributions = ['alpha', 'anglit', 'arcsine', 'argus', 
    #     'beta', 'betaprime', 'bradford', 'burr', 'burr12', 
    #     'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 
    #     'dgamma', 'dweibull', 
    #     'erlang', 'expon', 'exponnorm', 'exponpow', 'exponweib', 
    #     'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 
    #     'gamma', 'gausshyper', 'genexpon', 'genextreme', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'geninvgauss', 'genlogistic', 'gennorm', 'genpareto', 'gilbrat', 'gompertz', 'gumbel_l', 'gumbel_r', 
    #     'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm', 'hypsecant', 
    #     'invgamma', 'invgauss', 'invweibull', 
    #     'johnsonsb', 'johnsonsu', 
    #     'kappa3', 'kstwo', 'kstwobign', 
    #     'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'loguniform', 'lomax', 
    #     'maxwell', 'mielke', 'moyal',
    #     'nakagami', 'ncf', 'norm', 'norminvgauss', 
    #     'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 
    #     'rayleigh', 'reciprocal', 'rice', 
    #     'semicircular', 'skewcauchy', 'skewnorm', 
    #     't', 'truncexpon', 'truncnorm', 'tukeylambda', 
    #     'uniform', 
    #     'vonmises', 'vonmises_line', 
    #     'wald', 'wrapcauchy']

    distributions = ['alpha', 'arcsine', 'beta', 'betaprime', 'burr', 'cauchy', 'expon', 'f', 'fisk', 'foldcauchy', 'gamma', 'genexpon', 'genextreme', 'geninvgauss', 'genpareto', 'gilbrat', 'gumbel_r', 'halfcauchy', 'halfgennorm', 'halfnorm', 'invgamma', 'invgauss', 'invweibull', 'kappa3', 'ksone', 'laplace', 'lognorm', 'loguniform', 'lomax', 'ncf', 'norm', 'norminvgauss', 'pareto', 'powerlognorm', 'rayleigh', 'recipinvgauss', 'reciprocal', 'skewcauchy', 'truncexpon', 'wald']

    f = ft.Fitter(data, distributions=distributions) 
    f.fit()
    
    print(f.summary())
    
    plt.show()

def henze_zirkler_test():
    # https://online.stat.psu.edu/stat505/book/export/html/636
    # https://www.geeksforgeeks.org/how-to-perform-multivariate-normality-tests-in-python/
    
    # Read the databases into a list of dataframes (on the date and value columns)
    dbs = [pd.read_csv('Database/Caudal_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Conductividad_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Nitratos_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Oxigeno disuelto_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/pH_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Pluviometria_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Temperatura_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Turbidez_pro.csv', delimiter=';').iloc[:, 0:2]
    ]
    
    # Initialize the result with the first dataframe
    result = dbs[0]
    
    # Loop over the rest of the dataframes and merge them one by one
    for i in range(1,len(dbs)):
        result = pd.merge(result, dbs[i], on='date', how='inner', suffixes=("", f"_{i}"))
    
    # Test
    hz, p, normal = pg.multivariate_normality(result.iloc[:10000, 1:], alpha=0.05)

    # If p>0.05, the it is normal
    print('hz=', hz, 'p=', p, 'normal=', normal)

def stationary_test():
    
    """Implements the Dickey-Fuller test"""
    # (Step 3) https://www.geeksforgeeks.org/how-to-check-if-time-series-data-is-stationary-with-python/
    from statsmodels.tsa.stattools import adfuller

    data = pd.read_csv('database/Turbidez_pro.csv', delimiter=';')
    data = data.value

    res = adfuller(data)

    print('Augmneted Dickey_fuller Statistic: %f' % res[0]) # If p < 0.05 is stationary
    print('p-value: %f' % res[1])

data = pd.Series()

if __name__ == '__main__':

    pass

    # Statistical analysis
    # Maxim, Minim, Mean, Mode, Std, Variance, Q1, Q2, Q3, InQR = basic_stats(data) 
    # print('Maximum: {}; Minimum: {}; Mean: {}; Q1: {}; Q3: {}; IQR: {}'. format(Maxim, Minim, Mean, Q1, Q3, InQR))

    # Boxplot
    # outliersBoxplot = boxplot(varName, Q1, Q3, InQR)

    # Test de distribución normal
    #

files=["BEN_pro.csv", "CO_pro.csv", "MXIL_pro.csv", "NO_pro.csv", "NO2_pro.csv", "O3_pro.csv", "PM2.5_pro.csv", "SO2_pro.csv", "TOL_pro.csv"]

#stats = pd.DataFrame(columns=["variable", "estatística shapiro-wilk", "valor p shapiro-wilk", "estatística kolmogorov-smirnov", "valor p kolmogorov-smirnov"])

for f in files:
    varname = f.split("_")
    varname = varname[0]

    # Gráficas
    #BENCENO
    # Cargar os datos do arquivo BEN_pro.csv nun DataFrame
    ruta_arquivo = os.path.join("Database", "BEN_pro.csv")
    df = pd.read_csv(ruta_arquivo, delimiter=";")

    #Para converter todo a tipo numérico
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Imprimir os nomes das columnas para identificar o nome correcto da columna de datos de benceno
    #print(df.columns)

    # Crear o histograma utilizando Seaborn
    #sns.histplot(data=df, x="value", kde=True, color="green")

        # Q-Q
    nome_columna = "value"
    sm.qqplot(df[nome_columna], line='s')

    # Mostrar a gráfica
    #plt.xlabel("Valor")
    #plt.ylabel("Frecuencia")
    #plt.title("Histograma do BENCENO")
    
    plt.xlabel("Cuantís teóricos")
    plt.ylabel("Cuantís observados")
    plt.title("Gráfica Q-Q do BENCENO")

    plt.show()





    #CO
    ruta_arquivo = os.path.join("Database", "CO_pro.csv")
    df = pd.read_csv(ruta_arquivo, delimiter=";")

    #Para converter todo a tipo numérico
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    #print(df.columns)

    #sns.histplot(data=df, x="value", kde=True, color="green")

        # Q-Q
    nome_columna = "value"
    sm.qqplot(df[nome_columna], line='s')
    
    #plt.xlabel("Valor")
    #plt.ylabel("Frecuencia")
    #plt.title("Histograma do CO")
    
    plt.xlabel("Cuantís teóricos")
    plt.ylabel("Cuantís observados")
    plt.title("Gráfica Q-Q do CO")

    plt.show()
    
  




    #MXIL
    ruta_arquivo = os.path.join("Database", "MXIL_pro.csv")
    df = pd.read_csv(ruta_arquivo, delimiter=";")

    #Para converter todo a tipo numérico
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    #print(df.columns)

    #sns.histplot(data=df, x="value", kde=True, color="green")

        # Q-Q
    nome_columna = "value"
    sm.qqplot(df[nome_columna], line='s')

    #plt.xlabel("Valor")
    #plt.ylabel("Frecuencia")
    #plt.title("Histograma do M-XILENO")
    
    plt.xlabel("Cuantís teóricos")
    plt.ylabel("Cuantís observados")
    plt.title("Gráfica Q-Q do M-XILENO")

    plt.show()
    #plt.show()






    #NO
    ruta_arquivo = os.path.join("Database", "NO_pro.csv")
    df = pd.read_csv(ruta_arquivo, delimiter=";")

    #Para converter todo a tipo numérico
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    #print(df.columns)

    #sns.histplot(data=df, x="value", kde=True, color="green")

        # Q-Q
    nome_columna = "value"
    sm.qqplot(df[nome_columna], line='s')

    #plt.xlabel("Valor")
    #plt.ylabel("Frecuencia")
    #plt.title("Histograma do NO")
    
    plt.xlabel("Cuantís teóricos")
    plt.ylabel("Cuantís observados")
    plt.title("Gráfica Q-Q do NO")

    plt.show()
    #plt.show()


    #NO2
    ruta_arquivo = os.path.join("Database", "NO2_pro.csv")
    df = pd.read_csv(ruta_arquivo, delimiter=";")

    #Para converter todo a tipo numérico
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    #print(df.columns)

    #sns.histplot(data=df, x="value", kde=True, color="green")

        # Q-Q
    nome_columna = "value"
    sm.qqplot(df[nome_columna], line='s')
    
    #plt.xlabel("Valor")
    #plt.ylabel("Frecuencia")
    #plt.title("Histograma do NO2")
    
    plt.xlabel("Cuantís teóricos")
    plt.ylabel("Cuantís observados")
    plt.title("Gráfica Q-Q do NO2")

    plt.show()
    #plt.show()




    #O3
    ruta_arquivo = os.path.join("Database", "O3_pro.csv")
    df = pd.read_csv(ruta_arquivo, delimiter=";")

    #Para converter todo a tipo numérico
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    #print(df.columns)

    #sns.histplot(data=df, x="value", kde=True, color="green")

        # Q-Q
    nome_columna = "value"
    sm.qqplot(df[nome_columna], line='s')

    #plt.xlabel("Valor")
    #plt.ylabel("Frecuencia")
    #plt.title("Histograma do O3")
    
    plt.xlabel("Cuantís teóricos")
    plt.ylabel("Cuantís observados")
    plt.title("Gráfica Q-Q do O3")

    plt.show()
    #plt.show()



    #PM2.5
    ruta_arquivo = os.path.join("Database", "PM2.5_pro.csv")
    df = pd.read_csv(ruta_arquivo, delimiter=";")

    #Para converter todo a tipo numérico
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    #print(df.columns)

    #sns.histplot(data=df, x="value", kde=True, color="green")

        # Q-Q
    nome_columna = "value"
    sm.qqplot(df[nome_columna], line='s')

    #plt.xlabel("Valor")
    #plt.ylabel("Frecuencia")
    #plt.title("Histograma do PM2.5")
    
    plt.xlabel("Cuantís teóricos")
    plt.ylabel("Cuantís observados")
    plt.title("Gráfica Q-Q do PM2.5")

    plt.show()
    #plt.show()





    #SO2
    ruta_arquivo = os.path.join("Database", "SO2_pro.csv")
    df = pd.read_csv(ruta_arquivo, delimiter=";")

    #Para converter todo a tipo numérico
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    #print(df.columns)

    #sns.histplot(data=df, x="value", kde=True, color="green")

    
    # Q-Q
    nome_columna = "value"
    sm.qqplot(df[nome_columna], line='s')

    #plt.xlabel("Valor")
    #plt.ylabel("Frecuencia")
    #plt.title("Histograma do SO2")
    
    plt.xlabel("Cuantís teóricos")
    plt.ylabel("Cuantís observados")
    plt.title("Gráfica Q-Q do SO2")

    plt.show()
    #plt.show()




    #TOL
    ruta_arquivo = os.path.join("Database", "TOL_pro.csv")
    df = pd.read_csv(ruta_arquivo, delimiter=";") 

    #Para converter todo a tipo numérico
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    #print(df.columns)

    #sns.histplot(data=df, x="value", kde=True, color="green")


        # Q-Q
    nome_columna = "value"
    sm.qqplot(df[nome_columna], line='s')

    #plt.xlabel("Valor")
    #plt.ylabel("Frecuencia")
    #plt.title("Histograma do TOLUENO")
    
    plt.xlabel("Cuantís teóricos")
    plt.ylabel("Cuantís observados")
    plt.title("Gráfica Q-Q do TOLUENO")

    plt.show()
    #plt.show()

    df = pd.read_csv(f"Database/{f}", sep=";", encoding="utf-8")
    print(df)
    
        #Proba Shapiro-Wilk

    stat_sw, p_value_sw=shapiro(df[nome_columna])
    print("Estatística SW", stat_sw)
    print("Valor p SW", p_value_sw)

        #Kolmogorov-Smirnov Test

    stat_ks, p_value_ks=kstest(df[nome_columna], 'norm')
    print("Estatística KS", stat_ks)
    print("Valor p KS", p_value_ks)

   
    stats.loc[len(stats.index)] = [varname, stat_sw, p_value_sw, stat_ks, p_value_ks]

print(stats)

stats.to_csv("tests.csv", sep=";", encoding="utf-8", index=False)


