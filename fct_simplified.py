#fichier python contenant les calculs

import math
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
import sklearn
import statistics
from statistics import multimode, mode
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import fftpack
from scipy.integrate import cumulative_trapezoid
from scipy import LowLevelCallable
import codecs
from chardet.universaldetector import UniversalDetector
# Cette fonction calcule l'écart-type d'un interval de valeur [a;b[
def calculEcartType(a, b, listQ):
    sum = 0
    Qmoy = 0
    for i in range(a, b):
        Qmoy += listQ[i]
    Qmoy = Qmoy / (b - a)

    for i in range(a, b):
        sum += pow(listQ[i] - Qmoy, 2)
    return math.sqrt(sum / (b - a - 1))

# Cette fonction permet de calculer le débit par dérivation,
# Elle prend en argument l'intervalle [a,b[ dans lequel il calculera le débit
# de listA avec un intervale de temps constant dt.
def calculDebitDerivation(a, b, dt, listA):
    listQ = []
    for i in range(a, b -1):
        listQ.append((listA[i + 1] - listA[i]))# / dt)
    return listQ

import time
# convertit et échantillonne le fichier csv donné dans path en liste
def convertCSVToList(path, target='MASSE_BALANCE', n=1,sep=",",
                     headerFormat="2 lignes",
                     decimal=".",ignoredLines=0):
    # we try to detect the encoding of the file
    detector = UniversalDetector()
    i = 0
    for line in open(path, 'rb'):
        i += 1
        detector.feed(line)
        if detector.done: break
        if i > 100: break  # number of max lines that we sniff
    detector.close()
    encoding = detector.result['encoding']
    print(encoding)

    if target is None:

        useful_col = []
        useful_Units=[]

        with open(path,'r',encoding= encoding) as f:
            for i in range(ignoredLines):
                f.readline()
            first_line = f.readline().replace("\n","").split(sep)
            if headerFormat=="2 lignes":
                second_line=f.readline().replace("\n","").split(sep)
            sample_line=f.readline().replace("\n","").split(sep)

        for i in range(len(first_line)):
            if first_line[i] !="":
                try :
                    if len(sample_line[i]) >0:
                        if decimal==",":
                            float(sample_line[i].replace(",", "."))
                        else :
                            float(sample_line[i])
                        useful_col.append(first_line[i])
                    if headerFormat == "2 lignes":
                        useful_Units.append(second_line[i])
                except ValueError as e:
                    pass
        skipping=None
        if ignoredLines !=0:
            skipping = [x for x in range(ignoredLines)]
            if len(useful_Units) > 0 :
                skipping.append(skipping[-1]+2)
        elif len(useful_Units) > 0:
            skipping=[1]

        df = pd.read_csv(path, low_memory=False,
                             quoting=csv.QUOTE_NONNUMERIC,sep=sep,
                             encoding=encoding ,
                             decimal=decimal,dtype='float64', usecols=useful_col,
                             skiprows=skipping,error_bad_lines=False,warn_bad_lines=False)
        return df.T.values.tolist(), useful_Units if headerFormat=="2 lignes" else [""]*len(df.columns), useful_col

    else:
        df = pd.read_csv(path, usecols=[target], low_memory=False,quoting=csv.QUOTE_NONNUMERIC,
                             sep=sep,encoding=encoding,decimal=decimal)
        df = df.iloc[1:]
        dt = pd.to_numeric(df[target], downcast='float')

        dt = dt.iloc[::n]
        return dt.values.tolist()

# fait la moyenne glissante :
def moving_average(a, n=75):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# corriger l'offset d'une liste entière
def offsetCorrection(listA, offset=0):
    if offset == 0:
        return [x - listA[0] for x in listA]
    else:
        return [x - offset for x in listA]

def csvSniffer(path) :
    with open(path, 'r') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.readline())
        print(dialect.delimiter)
        secondLine=csvfile.readline().split(dialect.delimiter)
        thirdLine = csvfile.readline().split(dialect.delimiter)

#fait un filtrage par fenêtre glissante
def rollingFilter(w_length, w_type, listA):
    df = pd.DataFrame(data=listA,columns=['Values'])
    if w_type == 0:
        df_rolling = df.rolling(window=w_length, win_type=None).mean()
    elif w_type == 1:
        df_rolling = df.rolling(window=w_length, win_type='triang').mean()
    elif w_type == 2:
        df_rolling = df.rolling(window=w_length, win_type='gaussian').mean(std=w_length/2)
    return df_rolling

#calcule l'écart type glissant d'une courbe
def rollingStd(w_length, listA):
    df= pd.DataFrame(data= listA, columns= ['Values'])
    df_rolling= df.rolling(window=w_length).std()
    listf= df_rolling['Values'].tolist()
    return listf

def rollingStdNormalized(w_length, listA):
    stdDeviation= rollingStd(w_length, listA)
    #print(multimode(stdDeviation))
    #print(mode(stdDeviation))
    '''
    #pour afficher les poles qu'on veut avec le plus d'occurences
    #on crée un dict où on compte le nombre d'occurence d'une valeur
    counts= dict()
    for val in stdDeviation:
        counts[val]= counts.get(val,0)+ 1

    #on prend ensuite la clef qui est apparue le plus de fois
    bigcount= None
    bigname= None
    listOfTup=[]
    for k, v in counts.items():
        newtup=(v, k)
        listOfTup.append(newtup)
    listOfTup = sorted(listOfTup, reverse= True)
    print(listOfTup[:10])
    '''
    #on peut obtenir ici en une seule ligne le mode le plus petit
    std_min_mode= min(multimode(stdDeviation))
    stdDeviation_normalized= []
    for x in stdDeviation:
        if std_min_mode !=0:
            stdDeviation_normalized.append(x/std_min_mode)
        else:
            print('division par 0 impossible, pas de mode détecté')
            break
    return stdDeviation_normalized


# fait une fft et renvoie le spectre de fréquence ainsi que le signal filtré
def fourierFilter(yaxis, amplitude=50):
    yaxis=np.array(yaxis)
    four = fftpack.fft(yaxis)
    frq = fftpack.fftfreq(yaxis.size)
    frq_abs = np.abs(frq)
    power = np.abs(four)

    #on filtre notre spectre de fréquences
    four[power < amplitude] = 0

    #on calcule la fft inverse
    filtered_signal = fftpack.ifft(four)

    #on prend les val abs des signaux pour retirer les composantes complexes
    filtered_signal_abs = np.abs(filtered_signal)
    fourier_abs=np.abs(four)

    #on convertit en liste
    frequences= frq_abs.tolist()
    fourier= fourier_abs.tolist()
    filtered= filtered_signal_abs.tolist()

    return frequences, fourier, filtered

# normalisation MinMax
def MinMaxNormalisation(listA):
    X_list=np.array(listA)
    X_list=X_list.reshape(-1, 1)
    #on définit un scaler
    scaler=MinMaxScaler()
    #on applique ce scaler grâce à la méthode fit_transform
    X_normalized= scaler.fit_transform(X_list)
    X_normalized= X_normalized.flatten()
    listf=X_normalized.tolist()
    return listf

# normalisation z-score
def StandardNormalisation(listA):
    X_list = np.array(listA)
    X_list = X_list.reshape(-1, 1)

    scaler = StandardScaler()

    X_normalized = scaler.fit_transform(X_list)
    X_normalized = X_normalized.flatten()
    listf = X_normalized.tolist()
    return listf

# normalisation euclidienne (valeur abs)
def ValAbsNormalisation(listA):
    X= np.array(listA)
    X= X.reshape(-1,1)

    #pas de scaler ici, on utilise la fonction prepocessing.normalize en précisant la norme euclidienne 'l2'
    X_normalized = sklearn.preprocessing.normalize(X, norm='l2')
    X_normalized= X_normalized.flatten()
    listf= X_normalized.tolist()
    return listf

# normalisation moyenne
def MeanNormalisation(listA):
    X= listA
    X_moy= statistics.mean(X)
    X_max_min= max(X)- min(X)
    X_normalized= []
    #on crée notre nouvelle liste normalisée
    for x in X:
        X_normalized.append((x- X_moy)/X_max_min)
    return X_normalized

# calcule d'intégrale
def Integration(listA, inf, sup):
    x = np.linspace(inf, sup, len(listA))
    integral = cumulative_trapezoid(listA, x, initial=0)
    return max(integral)
if __name__ == "__main__":
    print(sys.argv[1])


    """paths=[r"concatenation.txt",r"Données 0x2007 0x0957 08_07_2021 15_49_07.csv",
           "bbb.csv"]


    A=convertCSVToList(paths[0],None,1,"\t",decimal=",",headerFormat="2 lignes")[0]
    if len(A) ==78 :
        print("Test 1 passé  Fichier Julie lu")
    else : print("Test 1 failed n=",len(A))

    print("\n---------\n")
    A=convertCSVToList(paths[1], None, 1, ";", decimal=".",headerFormat="1 ligne",ignoredLines = 12)[2]
    if len(A) ==9 :
        print("Test 2 passé  Fichier Florestan lu")
    else : print("Test 2 failed n=",len(A))

    print("\n---------\n")
    A=convertCSVToList(paths[2], None, 1, ",", headerFormat="2 lignes")[0]
    if len(A) ==1:
        print("Test 3 passé  Fichier centrale lu")
    else : print("Test 3 failed n=",len(A))

    print("finished")"""






























