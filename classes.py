import numpy
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import  QApplication, QTableWidgetItem, QLabel, QComboBox
from PyQt5.QtGui import QPixmap , QCursor
from PyQt5.QtCore import  pyqtSignal
import time



GLOB_FILE_CREATE = 0
GLOB_FILE_APPEND = 1
fileLogName = "log.txt"
separator = "----------------------------------\n"
mini_separator="\t\t---------\n"
GLOB_INV_GRAPH_INDEX=-1
GLOD_SOFT_NAME="Expert"
NOT_AVAILABLE="cette fonctionnalité n'est pas en service"

DO_REFRESH_CALCULATION=True
DO_NOT_REFRESH_CALCULATION=False
DO_REFRESH_DISPLAY=True
DO_NOT_REFRESH_DISPLAY=False
SHOW_TIME_NORMAL=5000
SEPARATOR_ENUM = {"Virgule": ",",
                  "Point": ".",
                  "Point-virgule": ";",
                  "Deux-points": ":",
                  "Tabulation": "\t",
                  }
TIMESTEP_ENUM={"0.1 ms":0.0001,
               "1 ms":0.001,
               "10 ms":0.01,
               "100 ms":0.1,
               "500 ms":0.5,
               "1 s":1,
               "5 s":5,
               "10 s":10,
               "20 s":20,
               "30 s":30,
               "1 min":60,
               "5 min":300,
               "10 min":600,
               "15 min":900,
               "20 min":1200,
               "30 min":1800,
               "1 h":3600,
               "1 jour":86400}


CONFIG_LIST=["CSV_sep","CSV_thous","CSV_header","CSV_timeStep","CSV_ignoreLines","Slice_moving_average",
             "Slice_with_ramp","Slice_with_slope","Slice_step_lim","Slice_step_precision",
             "Slice_ramp_precision","Slice_min_width","Slice_left_trunc","Slice_right_trunc"]
CONFIG_USER_NAME=["séparateur des champs","séparateur décimal","Nombre de ligne de l'en-tête",
                  "pas en temps","Nombre de points des moyennes mobiles","calcul des paliers",
                  "calcul palier avec filtrage des pentes","seuil filtre pente","précision plateau",
                  "précision rampe","nombre de points minimum par séquence","Troncature gauche",
                  "troncature droite"]
CONFIG_DICT=dict(zip(CONFIG_LIST,CONFIG_USER_NAME))
DV_MODE=["DV","Dérivée ","Kalman","Operation"]  #modification

#droite | gauche | haut | bas
TRANSLATE_OPERATOR=["Décalage Droit", "Décalage Gauche", "Décalage Haut",
                    "Décalage Bas","Multiplication","Division","Normalisation MinMax",
                    "Normalisation Z-Score", "Normalisation Val. Abs", "Normalisation Moyenne",
                    "Ecart-Type Glissant", "Ecart-Type Glissant Normalisé","Moyenne glissante"]
DEFAULT_KALMAN_VALUES=['1000','0.1','0.001','0.00001']
DEFAULT_KALMAN_FREQUENCY='1000'
DEFAULT_ROLLING_FILTER_PARAMS=['10',None] #modif

DATAFRAME_PRECISION=3
DEBUG=False

import Kalman
from PyQt5.QtCore import Qt
import pandas as pd
from contextlib import contextmanager
import fct_simplified
import PSI
import matplotlib.pyplot as plt

"""should be used like this :
with wait_cursor():
    # do lengthy process
    pass"""

@contextmanager
def wait_cursor():
    try:
        #put the wait Cursor
        QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        yield
    finally:
        QApplication.restoreOverrideCursor()

# write function
def WriteToDebugFIle(obj, iMode: int = GLOB_FILE_APPEND, fileName: str = fileLogName, iError: int = 0, stError: str = ""):
    if iMode == GLOB_FILE_CREATE:
        fichier = open(fileName, "w", encoding="utf-8")
        fichier.write(str(obj))
        fichier.close()
    elif iMode == GLOB_FILE_APPEND:
        fichier = open(fileName, "a", encoding="utf-8")
        fichier.write(str(obj))
        fichier.close()
    else:
        iError = 1

def ReadProjectFile(path):
    file1 = open(path, 'r', encoding='utf-8')
    version = file1.readline().split(":")[1]
    line_path=file1.readline().replace("\n","").split(":",1)[1]
    path_to_csv_file=""
    for x in line_path :
        path_to_csv_file+=x
    config =file1.readline().replace("\n","").split(":")[1].split(",")
    NfiltParam=int(file1.readline().split(":")[1])
    ListOfParam=[]
    for i in range(NfiltParam):
        line=file1.readline().split(":")
        name=line[0]
        B=line[1].split(",")
        param=[float(x) for x in B[1:]]
        forme=B[0]
        ListOfParam.append([name,forme,param])
    return version,path_to_csv_file,config,ListOfParam
def ReadProjectFileToDict(path):
    try :
        listOfLines = open(path, 'r', encoding='utf-8').read().split("\n")
    except UnicodeDecodeError as ue:
        print(ue)
        return None
    projectDict={}
    for x in DV_MODE :
        projectDict[x]=[]

    for line in listOfLines : #on lit ligne par ligne, les clefs du dico sont les premiers mots de chaque ligne
        entries=line.split(":",1)
        if len(entries)>1 :
            key =entries[0]
            values=entries[1]
            projectDict  [key]=values
        else :
            entries=entries[0].split(",")
            if entries[0]!="":
                projectDict[entries[0]].append(entries[1:])
    return (projectDict)
def removeObjFromList(obj:object,listA:list):
    index=listA.index(obj)
    return listA.pop(index),index
def createQPixMap(path):
    pixmap = QPixmap(path)
    label = myQLabel()
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)
    return label
def listPrinter(A):
    s=""
    for x in A:
        s+= str(x) +","
    return s[:-1]
def dictPrinter(A):
    SSS=""
    for k,v in A.items():
        SSS+=str(k)+","+str(v)
    return SSS
#----------------------------------------------------------------------------------------------------
#Méthodes implémentées pour gérer le traitement automatique
def changeFileCsvPath(path, newCsv):
    try:
        listOfLines = open(path, 'r', encoding='utf-8').readlines()
    except UnicodeDecodeError as ue:
        print(ue)
        return None
    listOfLines[1]= "path_to_csv_file:"+newCsv+"\n"
    return listOfLines

# ---------------------------------------------------------------------------------------------------
#Ne sert qu'à stocker les paramètres du filtre de Kalman/Moyenne glissante
class DataKalm:
    # kalman or STDNV with param =[q0,q1,r0,r1]
    #TODO : change forme to be integer
    def __init__(self, arrParm: list=None, filtre:str=None, amplitude:str= None, longueur: str= None, type: str=None, forme:str=None,
                 freq:str=None, arrStepValues: list=None,arrStdValues: list=None):
        self.filtre= filtre
        self.amplitude= amplitude
        self.forme= forme
        self.longueur= longueur
        self.type= type
        self.freq= freq
        self.arrParm = arrParm
        self.arrStepValues= arrStepValues
        self.arrStdValues= arrStdValues
    def __str__(self):
        SSS = "DataKalm :\n"
        SSS+= ("Filtre :" + self.filtre + "\n")
        SSS+= ("Amplitude:" + self.amplitude+ "\n")
        SSS+= ("Forme :" + self.forme + "\n")
        SSS += ("Longueur :" + self.longueur + "\n")
        SSS += ("Type :" + self.type + "\n")
        SSS+= ("Frequence :" + self.freq + "\n")
        SSS += ("paramètres :" + str(self.arrParm) + "\n")
        SSS += ("StepValues :" + str(self.arrStepValues) + "\n")
        SSS += ("STDValues :" + str(self.arrStdValues) + "\n")
        SSS += separator
        return SSS

class DataDeriv:
    def __init__(self,nPoints):
        self.nPoints=nPoints
        
    def __str__(self):
        return str(self.nPoints)

class DataOp:
    def __init__(self, typeop:str, decalage:str):
        self.typeop=typeop
        self.decalage=decalage
    def __str__(self):
        SSS = "Operation :\n"
        SSS += ("Destination :" + self.destination + "\n")
        SSS += ("Type d'opération :" + self.typeop + "\n")
        SSS += ("Décalage :" + self.decalage + "\n")
        SSS += separator
        return SSS

# ---------------------------------------------------------------------------------------------------
class DataVari:
    def __init__(self, stName: str, arValu: list, stUnit: str = "", stLabl: str = "",index=None,
                 arDataVari:list=None,obSpec: DataKalm = None,obAxes =None,obLine=None,parent=None, x_axis=None, x_value=None):
        self.stName = stName
        self.arValu = arValu
        self.stUnit = stUnit
        if stLabl ==  "":
            stLabl=stName+" "+stUnit
        self.stLabl = stLabl
        if arDataVari is None:
            self.arDataVari=[]
        else:
            self.arDataVari=arDataVari
        if index ==None:
            self.index=0
        else :
            self.index=index

        # x_axis gère l'axe des abscisses choisi pour tracer le graphe
        if x_axis is None:
            self.x_axis = 'DEFAUT'
        if x_value is None:
            self.x_value = 0
        else:
            self.x_axis = x_axis

        self.obSpec = obSpec
        self.obAxes=obAxes
        self.obLine=obLine
        self.bDoStatCalc=DO_NOT_REFRESH_CALCULATION
        self.bDoGrafDisp=DO_NOT_REFRESH_DISPLAY
        self.parent=parent
    def changeName(self,name,label=None):
        self.stName=name
        if label ==None:
            self.stLabl=name+" "+self.stUnit
        else :
            self.stLabl=label

        try :
            self.obLine.set_label(self.stLabl)
            self.obAxes.legend()
        except:
            pass
    #gère la normalisation
    def NormalizeMinMax(self):
        fct_simplified.MinMaxNormalisation(self.arValu)
    #gère la FFT (filtre est commun aux trois opérations de filtrages, cela indique le type de filtrage (son index)
    def DoFFT(self, filtre, amplitude):
        #fréquences et fourier sont utilisés pour créer le spectre
        #signal_filtré crée juste le signal filtré que l'on va plot dans le PlotterWidget
        frequences, fourier, signal_filtre = fct_simplified.fourierFilter(self.arValu, amplitude)
        obSpec = DataKalm(filtre=filtre, amplitude= amplitude)
        dataFFT= DataVari("Filtre_FFT " + self.stName, signal_filtre, parent=self, obSpec=obSpec,index=self.index)
        self.arDataVari.append(dataFFT)
        index_dataFFT= self.arDataVari.index(dataFFT)
        return frequences, fourier, index_dataFFT
    def RedoFFT(self, indexFFT, amplitude):
        frequences, fourier, signal_filtre = fct_simplified.fourierFilter(self.arValu, amplitude)
        self.arDataVari[indexFFT].updateValues(signal_filtre, amplitude, noRelim=True)
        return frequences, fourier

    #gère le 'rolling filter'
    def DoRollingFilter(self, filtre, win_length, win_type):
        offset= 0.5*win_length
        df_filt_list= fct_simplified.rollingFilter(win_length, win_type, self.arValu)
        filt_list= df_filt_list['Values'].tolist()
        obSpec= DataKalm(longueur=win_length, type=win_type, filtre=filtre)
        dataRF= DataVari("Moyenne_Glissante " + self.stName, filt_list, parent=self,obSpec=obSpec, index= self.index-offset)
        self.arDataVari.append(dataRF)
        index_dataRF= self.arDataVari.index(dataRF)
        return index_dataRF

    def RedoRollingFilter(self, win_length, win_type, indexRF):
        offset = 0.5 * win_length
        df_filt_list = fct_simplified.rollingFilter(win_length, win_type, self.arValu)
        filt_list = df_filt_list['Values'].tolist()
        self.arDataVari[indexRF].updateValues(filt_list,longueur=win_length, type=win_type, offset=-offset,noRelim=True)
    # Calcule le filtre de Kalman avec le jeu de données
    # STDNV et et kalman sont stockés dans arDataVari !
    def DoKalmanOperation(self,filtre, forme, arParm: list = None,dt:float=0.001,freq:int=1):
        if arParm is None:
            arParm = [float(x) for x in DEFAULT_KALMAN_VALUES]
            freq = DEFAULT_KALMAN_FREQUENCY
        [r0, r1, q0, q1] = arParm
        temp_list = Kalman.GetKalmanResults(self.arValu, QRef0=q0, QRef1=q1, RRef0=r0, RRef1=r1,dTimeStep=dt,forme=forme,freq=freq)
        obSpec=DataKalm(arrParm= arParm, filtre=filtre, forme=forme, freq=freq)
        dataK= DataVari("KALMAN " + self.stName, temp_list[0], self.stUnit[:-1] + "/h]", parent=self,obSpec=obSpec,index=self.index)
        self.arDataVari.append(dataK)
        dataSTDNV= DataVari("STDNV " + self.stName, temp_list[-1], parent=self,obSpec=obSpec,index=self.index)
        self.arDataVari.append(dataSTDNV)
        dataStep= DataVari("PROFIL " + self.stName, temp_list[1], parent=self,obSpec=obSpec,index=self.index)
        self.arDataVari.append(dataStep)
        index_dataK = self.arDataVari.index(dataK)
        index_dataSTDNV = self.arDataVari.index(dataSTDNV)
        index_dataStep = self.arDataVari.index(dataStep)
        return index_dataK, index_dataSTDNV, index_dataStep
    def RedoKalmanOperation(self,forme,indexK, indexSTDNV, indexStep, arParm:list=None,dt:float=0.001,freq:int=1):
        if arParm is None:
            arParm = [float(x) for x in DEFAULT_KALMAN_VALUES]
            freq = DEFAULT_KALMAN_FREQUENCY
        [r0, r1, q0, q1] = arParm

        temp_list = Kalman.GetKalmanResults(self.arValu, QRef0=q0, QRef1=q1, RRef0=r0, RRef1=r1,dTimeStep=dt,forme=forme,freq=freq)

        self.arDataVari[indexK].updateValues(temp_list[0],freq=freq,arParm=arParm,arrStepValueslist=temp_list[2],arrStdValueslist=temp_list[3], noRelim=True)
        self.arDataVari[indexSTDNV].updateValues(temp_list[-1],freq=freq,arParm= arParm, noRelim=True)
        self.arDataVari[indexStep].updateValues(temp_list[1], freq=freq,arParm=arParm, noRelim=True)
    def derivation(self,dt,n=1000):
        A= fct_simplified.calculDebitDerivation(0,len(self.arValu),dt,self.arValu)
        A=[x*3600000 for x in A]
        B= fct_simplified.moving_average(A,n)
        return B.tolist()
    def integrate(self, inf, sup):
        value = fct_simplified.Integration(self.arValu, inf, sup)
        return value
    def getStats(self, dt=None):
        # cette fonction renvoie l'intervalle de valeurs à étudier de la courbe
        if self.obAxes != None:
            start, end = self.obAxes.get_xlim()
        else:
            return self.arValu
        # si on a un autre axe des abscisses que 'DEFAUT' on adapte pour avoir en sortie
        # le start et end comme borne de l'intervalle des données en y
        if self.x_axis != 'DEFAUT':
            #y_data = self.obLine.get_ydata()
            x_data = self.obLine.get_xdata()
            # on tri notre axe des abscisses (comme sur le graphe)
            x_data.sort()
            # on récupère les index de start et end
            i=0
            for x in x_data:
                if x >= start:
                    pos_start = i
                    break
                i = i + 1
            n = 0
            for x in x_data:
                if x >= end:
                    pos_end = n
                    break
                n = n + 1
            start = pos_start
            end = pos_end
        return self.arValu[int(start) if int(start) >= 0 else 0:int(end)]

    def getxLim(self, dt=None):
            start, end= self.obAxes.get_xlim()
            if self.x_axis != 'DEFAUT':
                x_data = self.obLine.get_xdata()
                # on tri notre axe des abscisses (comme sur le graphe)
                x_data.sort()
                # on récupère les index de start et end
                i = 0
                for x in x_data:
                    if x >= start:
                        pos_start = i
                        break
                    i = i + 1
                n = 0
                for x in x_data:
                    if x >= end:
                        pos_end = n
                        break
                    n = n + 1
                start = pos_start
                end = pos_end
            return start, end
    def updateValues(self,val,amplitude: float=None, longueur: float=None, type:str=None,
                     freq:str=None,arParm:list=None,arrStepValueslist=None,arrStdValueslist=None,offset=None,noRelim=False):
        self.arValu=val
        if arParm is not None:
            self.obSpec.arrParm=arParm
            if freq is not None:
                self.obSpec.freq=freq
        if longueur is not None:
            self.obSpec.longueur=longueur
        if amplitude is not None:
            self.obSpec.amplitude=amplitude
        if type is not None:
            self.obSpec.type=type
        if arrStepValueslist is not None:
            self.obSpec.arrStepValueslist=arrStepValueslist
        if arrStdValueslist is not None:
            self.obSpec.arrStdValueslist=arrStdValueslist
        self.obLine.set_ydata(self.arValu)
        if offset is None:
            self.obLine.set_xdata([i for i in range(len(self.arValu))])
        if offset is not None:
            self.obLine.set_xdata([(i+offset) for i in range(len(self.arValu))])
        if self.obAxes != None and not noRelim:
            self.relim()
    def getParent(self):
        return self.parent
    def changeAX(self,newAX):
        # we remove the legend of the 1st graph
        if self.obAxes != None and self.obLine != None:
            self.obLine.set_label( '_nolegend_')
            self.obAxes.legend()

            #clear the line
            self.obLine.remove()
            #relim the previous ax
            self.relim()


        #we then plot the new line
        self.obAxes=newAX
        self.plot(Ax=self.obAxes)
        #self.obLine,=self.obAxes.plot(self.arValu,label=self.stLabl,picker=5)
        #self.obAxes.legend()

        # we relim the new ax
        #self.relim()
    def changeColor(self,hex):
        self.obLine.set_color(hex)
        for line in self.obAxes.legend().legendHandles :
            if self.obLine== line:
                line.set_color(hex)
                break
    def relim(self):
        try :
            self.obAxes.relim()
            self.obAxes.autoscale_view()
            #self.obAxes.set_xlim(self.index,self.index+len(self.arValu))
        except Exception as e:
            print("Relim failed! ")
            print(e)
    def removePlot(self): #removePlot s'occupe de retirer le tracé de la variable, pas le graphe
        if self.obAxes != None:
            self.obLine.set_label('_nolegend_')
            self.obAxes.legend()
            try :
                # clear the line
                self.obLine.remove()

                #relim
                self.relim()
                self.obAxes=None
            except :
                pass
    def clear(self):
        self.removePlot()
    def plot(self, Ax=None, noRelim=False):
        if Ax is not None:
            self.obAxes=Ax
        if self.obAxes is not None and self.x_axis == 'DEFAUT':
            self.obLine, =self.obAxes.plot([(self.index+x) for x in range(len(self.arValu))],
                                          self.arValu,label=self.stLabl,picker=5)
            self.obAxes.legend()
            if noRelim ==False: #quand on ajoute une courbe ça ajuste l'échelle sur cette courbe ajoutée mais ce n'est pas ce qui gère l'affichage de l'axe vertical.
                # relim
                self.relim()
        if self.obAxes is not None and self.x_axis != 'DEFAUT':
            # on crée le vecteur pour l'axe des abscisses
            self.obLine, = self.obAxes.plot(self.x_value, self.arValu, label=self.stLabl, picker=5)
            self.obAxes.legend()
            if noRelim == False:
                self.relim()

    def selectivePlot(self,a:int,b:int):
        if a > len(self.arValu)-2 :
            return True
        else:
            A=self.arValu[a:min(b,len(self.arValu))]
            self.obLine.set_ydata(A)
            self.obLine.set_xdata([ (self.index+x) for x in range (a,min(b,len(self.arValu)))])
            return False

    def __len__(self):
        return len(self.arValu)

    def __str__(self):
        SSS = ""
        if self.obSpec is not None:
            SSS += str(self.obSpec)
            SSS = SSS[:SSS.rfind('\n')]
            SSS = SSS[:SSS.rfind('\n')] + "\n"

        SSS += "DataVari :\n"
        SSS += (self.stLabl + "\n")
        SSS += ("values :\n" + str(self.arValu) + "\n")
        if self.arDataVari is not None:

            for i in range(len(self.arDataVari)):
                SSS += mini_separator
                SSS+="|arDataVari n°"+str(i)+"|"
                SSS+= str(self.arDataVari[i])
                SSS = SSS[:SSS.rfind('\n')]
                SSS = SSS[:SSS.rfind('\n')] + "\n"

        SSS += separator
        return SSS

    def translate(self,operation,name,value):
        try :
            if operation ==TRANSLATE_OPERATOR[0]:
                value=float(value)
                return DataVari(name,self.arValu,self.stUnit,name+" "+self.stUnit,value+self.index, obSpec= DataOp(0, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[1]:
                value = float(value)
                return DataVari(name,self.arValu,self.stUnit,name+" "+self.stUnit,-value+self.index, obSpec= DataOp(1, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[2]:
                value = float(value)
                return DataVari(name,[value+x for x in self.arValu],self.stUnit,name+" "+self.stUnit,self.index, obSpec= DataOp(2, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[3]:
                value = float(value)
                return DataVari(name,[-value+x for x in self.arValu],self.stUnit,name+" "+self.stUnit,self.index, obSpec= DataOp(3, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[4]:
                value = float(value)
                return DataVari(name,[value*x for x in self.arValu],self.stUnit,name+" "+self.stUnit,self.index, obSpec= DataOp(4, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[5]:
                value = float(value) if float(value) !=0 else 1e-10
                return DataVari(name,[x/value for x in self.arValu],self.stUnit,name+" "+self.stUnit,self.index, obSpec= DataOp(5, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[6]:
                liste= fct_simplified.MinMaxNormalisation(self.arValu)
                return DataVari(name, liste, self.stUnit, name+" "+self.stUnit,self.index, obSpec= DataOp(6, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[7]:
                liste= fct_simplified.StandardNormalisation(self.arValu)
                return DataVari(name, liste, self.stUnit, name+" "+self.stUnit, self.index, obSpec= DataOp(7, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[8]:
                liste= fct_simplified.ValAbsNormalisation(self.arValu)
                return DataVari(name, liste, self.stUnit, name+" "+self.stUnit, self.index, obSpec=DataOp(8, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[9]:
                liste= fct_simplified.MeanNormalisation(self.arValu)
                return DataVari(name, liste, self.stUnit, name+" "+self.stUnit, self.index, obSpec=DataOp(9, value), parent=self)
            if operation ==TRANSLATE_OPERATOR[10]:
                value = int(value)
                liste= fct_simplified.rollingStd(value, self.arValu)
                return DataVari(name, liste, self.stUnit, name + " " + self.stUnit, self.index, obSpec=DataOp(10, value),
                                parent=self)
            if operation ==TRANSLATE_OPERATOR[11]:
                value = int(value)
                liste= fct_simplified.rollingStdNormalized(value, self.arValu)
                return DataVari(name, liste, self.stUnit, name + " " + self.stUnit, self.index, obSpec=DataOp(11, value),
                                parent=self)
            if operation ==TRANSLATE_OPERATOR[12]:
                liste= fct_simplified.rollingFilter(2000, 0, self.arValu)
                listef= liste['Values'].tolist()
                return DataVari(name, listef, self.stUnit, name + " " + self.stUnit, self.index,
                                obSpec=DataOp(12, value),parent=self)
        except ValueError as e:
            print (e)
            return None


# ---------------------------------------------------------------------------------------------------

class DataGlob:
    def __init__(self, arData: list = [], arView: list = [], arFilt: list= [],arDeriv:list=[],
                 arCombi:list=[],arFig:list=[],project_name:str="NoName",path:str="",version:str=""):
        self.arData = arData
        self.arView = arView
        self.arFilt = arFilt
        self.project_name=project_name
        self.path_to_csv_file=path
        self.version=version
        self.arDeriv=arDeriv
        self.arCombi=arCombi

        self.Slices=ListOfSlices([],[])
        #stock la configuration
        self.arrConfig=[]
        self.arFig=arFig

    def DoKalmanOperation(self, index, arrParm: list):
        self.arData[index].DoKalmanOperation(arrParm)

    def PlotSlices(self,state):
        if state == Qt.Checked or state==True:
            self.Slices.plot()
        else:
            self.Slices.clear()

    def clearArray(self):
        self.arData.clear()
        self.arView.clear()
        self.arFilt.clear()
        self.arDeriv.clear()
        self.arCombi.clear()
        self.Slices = ListOfSlices([], [])

    def findDVByName(self,name):
        for x in self.arData:
            if x.stName == name:
                return x
        for x in self.arView:
            if x.stName == name:
                return x
        for x in self.arFilt:
            if x.stName == name:
                return x
        return None

    # TODO : record the AX
    def __str__(self):
        SSS= "version:"+self.version+"\n"
        SSS+="path_to_csv_file:"+self.path_to_csv_file+"\n"
        SSS+="Figure_Number:"+str(len(self.arFig))+"\n\n"

        for param,value in zip(CONFIG_LIST, self.arrConfig):
            SSS+= param+":"+str(value)+"\n"

        SSS+="Slices:"+str(self.Slices)+"\n"
        SSS+="\n"

        for DV in self.arView:
            fig=0
            cote=0
            if DV.obAxes != None:
                i=1
                for (ax1,ax2) in self.arFig:
                    if DV.obAxes == ax1 :
                        fig = i
                        cote = 0
                        break
                    elif DV.obAxes ==ax2:
                        fig=i
                        cote=1
                        break
                    i += 1
            if type(DV.obSpec) == type(None):
                SSS += DV_MODE[0]+","+str(fig)+","+str(cote)+"," + DV.stName   +"\n"
            elif type(DV.obSpec)== DataDeriv :
                SSS +=DV_MODE[1]+","+str(fig)+","+str(cote)+","  + DV.parent. stName+","+str(DV.obSpec)+"\n"
            elif type(DV.obSpec) == DataKalm:
                if DV.obSpec.filtre== 0:
                    SSS += DV_MODE[2]+","+str(fig)+","+str(cote)+","  + DV.parent. stName+","+ DV.stName+","\
                        +str(DV.obSpec.filtre)+","+str(DV.obSpec.forme)+","\
                        +str(DV.obSpec.freq)+","+listPrinter(DV.obSpec.arrParm)+"\n"
                if DV.obSpec.filtre== 1:
                    SSS += DV_MODE[2]+","+str(fig)+","+str(cote)+","  + DV.parent. stName+","+ DV.stName+","\
                        +str(DV.obSpec.filtre)+","+str(DV.obSpec.type)+","\
                        +str(DV.obSpec.longueur)+"\n"
                if DV.obSpec.filtre== 2:
                    SSS += DV_MODE[2] + "," + str(fig) + "," + str(
                        cote) + "," + DV.parent.stName + "," + DV.stName + "," \
                           + str(DV.obSpec.filtre) + "," \
                           + str(DV.obSpec.amplitude) + "\n"
            elif type(DV.obSpec) == DataOp:
                SSS += DV_MODE[3]+","+str(fig)+","+str(cote)+","+ DV.parent. stName+","+ DV.stName+","\
                       +str(DV.obSpec.typeop)+","+str(DV.obSpec.decalage)+"\n"

        return SSS

class ListOfSlices:
    def __init__(self, Slices, Axes, name: str= ""):
        self.Slices= Slices
        self.Axes= Axes
        self.listOfLines= []
        self.name= name

    def plot(self):
        for slice in self.Slices:
            for ax in self.Axes :
                a,b,moy= slice.dRighBound,slice.dLeftBound,slice.dAvg
                self.listOfLines.append( ax[0].axline((a, moy), (a, moy+.1),color='blue',alpha=0.3 ) )
                self.listOfLines.append( ax[0].axline((b, moy), (b, moy+.1),color='red',alpha=0.3) )
        print(len(self.Slices))


    def PlotOnLastAx(self):
        ax = self.Axes[-1][0]
        for slice in self.Slices:
            a, b, moy = slice.dRighBound, slice.dLeftBound, slice.dAvg
            self.listOfLines.append(ax.axline((a, moy), (a, moy + .1), color='blue', alpha=0.3))
            self.listOfLines.append(ax.axline((b, moy), (b, moy + .1), color='red', alpha=0.3))


    def clear(self):
        for x in self.listOfLines:
            try :
                x.remove()
            except:
                pass
    def restoreListOfSlices(self,block):
        self.name = block[0]
        block=block[1:]
        for triplet in block :
            triplet=triplet.split("|")
            PSId = PSI.ffi.new("struct PSI_SLICE_STRUCT *")
            PSId.dLeftBound=PSI.ffi.cast("double",float(triplet[0]))
            PSId.dRighBound=PSI.ffi.cast("double",float(triplet[1]))
            PSId.dAvg=PSI.ffi.cast("double",float(triplet[2]))
            self.Slices.append(PSId)

    def __str__(self):
        return self.name +","+ listPrinter([ str(x.dLeftBound)+"|"+str(x.dRighBound)+"|"+str(x.dAvg) for x in self.Slices]) if len(self.Slices)>0 else ""

    def __len__(self):
        return len(self.Slices)
# ---------------------------------------------------------------------------------------------------
#class fille de QTableWidgetItem qui nous permet de restorer les anciennes valeurs
class MyQTableWidgetItem(QTableWidgetItem):
    def __init__(self,text):
        self.oldValue=text
        super().__init__(text)

    def Cancel(self,obj=None):
        self.setText(self.oldValue)
    def Save(self,obj=None):
        self.oldValue=self.text()
#class fille de QLabel qui nous permet de cliquer dessus
class myQLabel(QLabel):
    clicked=pyqtSignal(object)
    def __init__(self):
        self.QTWI =None
        super().__init__()

    def setQTableWidgetItem(self,obj=None):
        self.QTWI=obj

    def mousePressEvent(self, ev):
        if self.QTWI is  None:
            self.clicked.emit(self)
        else:
            self.clicked.emit(self.QTWI)
#class fille de QAbstractTableModel qui nous permet de définir un model compatible avec les dataFrames
class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):

        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

        elif role == Qt.TextAlignmentRole:
            value = self._data.iloc[index.row()][index.column()]
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                # Align right, vertical middle.
                return Qt.AlignVCenter + Qt.AlignCenter

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])
#class fille de QComboBox qui nous permet de restorer les anciennes valeurs
class MyComboBox(QComboBox):
    popupAboutToBeShown = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super(QComboBox, self).__init__(parent)

        self.lastSelected = 0

    def showPopup(self):
        self.popupAboutToBeShown.emit()
        super(MyComboBox, self).showPopup()

    def save(self):
        self.lastSelected=self.currentIndex()

    def cancel(self):
        self.setCurrentIndex(self.lastSelected)
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setCentralWidget(self.table)
class MyQLineEdit(QtWidgets.QLineEdit):
    def __init__(self,parent=None):
        super(QtWidgets.QLineEdit, self).__init__(parent)
        self.savedText = ""


    def save(self):
        self.savedText = self.text()

    def cancel(self):
        self.setText(self.savedText)

