import matplotlib
import numpy
import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import fct_simplified
import math
from statistics import mode
from classes import SHOW_TIME_NORMAL, TIMESTEP_ENUM, wait_cursor, ListOfSlices, DEBUG, DATAFRAME_PRECISION,time
import PSI
import numpy as np
from scipy import fftpack


#handle multiple axis coordinates
def make_format(current, other):
    # current and other are axes
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x,y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Gauche: {:<20}    Droite: {:<}'
                .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
    return format_coord

def line_picker(line, mouseevent):
        """
        Find the points within a certain distance from the mouseclick in
        data coords and attach some extra attributes, pickx and picky
        which are the data points that were picked.
        """
        if mouseevent.xdata is None:
            return False, dict()
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        maxd = 1
        d = numpy.sqrt(
            (xdata - mouseevent.xdata) ** 2 + (ydata - mouseevent.ydata) ** 2)

        ind, = numpy.nonzero(d <= maxd)
        if len(ind):
            pickx = xdata[ind]
            picky = ydata[ind]
            props = dict(ind=ind, pickx=pickx, picky=picky)
            return True, props
        else:
            return False, dict()

class SnaptoCursor(object):
    def __init__(self, ax, x, y, hold=None):
        self.x = x
        self.y = y
        self.ax = ax
        self.xcoords = 0
        self.ycoords = 0
        self.results = ax.text(0.7, 0.9, '', size='large', bbox=dict(facecolor='white', alpha=0.5), zorder=10)
        if hold==None:
            self.lx = ax.axhline(y=min(y), color='k')  # the horiz line
            self.ly = ax.axvline(x=0, color='k')  # the vert line
            self.marker, = ax.plot([0],[min(y)], marker="o", color="crimson", zorder=3)
            self.txt = ax.text(0.7, 0.9, '', size='medium') #bulle info de texte

        # crée un curseur transparent et bloque la position du précédent
        else:
            self.lx = ax.axhline(y=min(y), color='w', zorder=0)  # the horiz line
            self.ly = ax.axvline(x=0, color='w', zorder=0)  # the vert line
            self.marker, = ax.plot([0], [min(y)], marker="o", color="w", zorder=0)
            self.txt = ax.text(0, 0, '', size='medium', zorder=0)  # bulle info de texte

    def mouse_move(self, event):

        if not event.inaxes:   return  #renvoie rien si on est pas dans la figure

        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0] #pour nous ça renvoie la valeur de l'abscisse de notre souris

        #searchsorted: returns an index or indices that suggest where x should be inserted
        #so that the order of the list self.x would be preserved
        #It returns the index i which satisfies: a[i-1]<v[i]<a[i+1]

        try:
            x = self.x[indx]
        except:
            pass
        try:
            y = self.y[indx]
        except:
            pass
        self.ly.set_xdata(x)
        self.lx.set_ydata(y)
        self.marker.set_data([x],[y])
        self.txt.set_text('x=%1.3f, y=%1.3f' % (x, y))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle() #redraw the current figure
        self.xcoords= x
        self.ycoords= y

    def measurement(self, list, step, freq):
        deltaT = np.abs(list[2] - list[0])
        deltaTs = deltaT*step
        deltay = np.abs(list[3]-list[1])
        dephasage= ((deltaTs)*freq)*360

        return deltaT, deltaTs, deltay, dephasage


class MplCanvas(QWidget):
    CursorsIsActive = 0
    InfoTable = pyqtSignal(object)
    FourierTable = pyqtSignal(object)
    def __init__(self, parent=None):
        QWidget.__init__(self,parent)
        self.canvas = FigureCanvas(Figure())

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        self.Ax=[] # -> maintenant de la forme [(ax,ax2)]
        self.canvas.axes = self.canvas.figure.add_subplot(111,picker=5) #définit la taille de la toile(canvas), pour picker il doit avoir un rôle
                                                    # dans le curseur et l'affichage de ses coordonnées
        self.Ax.append( (self.canvas.axes,self.canvas.axes.twinx())) #ajout des axes twins un en // avec y l'autre se superpose à x(il est caché)
        self.Ax[0][0].set_zorder( self.Ax[0][1].get_zorder() + 1) #prend la position (sur l'écran de Ax[0][1] et le met au dessus (en rajoutant 1)
                                                                #probablement pour cacher l'axe ajouté.
        self.Ax[0][0].set_facecolor("white")
        self.Ax[0][1].tick_params(axis='y', labelcolor="white",color ="white") #on cache l'axe des ordonnées (valeurs avec labelcolors) de droite et les ticks de graduations (colors)
        self.Ax[0][1].tick_params(labelbottom=True,labeltop=False,labelleft=False,labelright=True)
        #right = True, left = True : indique que les traits doivent à la fois être sur la ligne verticale de gauche et celle de droite (pour l'axe des x, c'est bottom = True, top = True).

        self.Ax[0][1].format_coord = make_format(self.Ax[0][1], self.Ax[0][0])
        self.nLigne = 1 #on affiche d'abord un graph
        self.canvas.axes.autoscale_view()
        self.setLayout(vertical_layout)
        self.toolbar = NavigationToolbar(self.canvas, self)
        vertical_layout.addWidget(self.toolbar) #la box vertical_layout contient le canvas + toolbox


        self.canvas.axes.plot([0]*10,picker=5)

        self.canvas.mpl_connect("button_press_event",self.onclick)

        self.annot = self.canvas.axes.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->")) #annot à les coordonnées des points où on clique et le texte qui s'affiche
        self.annot.set_visible(False)

        self.Picked_Flag = False
        self.canvas.mpl_connect("pick_event",self.onpick)

        self.interval=10
        self.step=10
        self.ptr=0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)

        self.canvas.figure.tight_layout()
        self.nbOfCursors= 0
        self.CursorsIsActive = 2
        self.listOfCursors= []

    def addSubPlot(self):
        if self.nLigne<4: #limite le nombre max de graph à 4
            QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
            self.nLigne += 1
            gs = gridspec.GridSpec(self.nLigne, 1,figure=self.canvas.figure)
            dt = self.Main.configWidget.timestepComboBox.currentText()[1:]

            for i in range (len(self.Ax)):
                self.Ax[i][0].set_subplotspec(gs[i,:])
                self.Ax[i][1].set_subplotspec(gs[i, :])
                self.Ax[i][0].set_xlabel(dt)

            self.toolbar.update()
            # add a new subplot - ajoute le premier subplot, avec deux axes des ordonnéees partageant un même axe x
            a =self.canvas.figure.add_subplot(gs[-1,:],picker=5)

            self.Ax.append( (a,a.twinx()))
            self.Ax[-1][1].format_coord = make_format(self.Ax[-1][1], self.Ax[-1][0])
            self.Ax[-1][0].set_zorder( self.Ax[0][0].get_zorder() )
            self.Ax[-1][0].set_facecolor("white")

            self.Ax[-1][0].set_xlabel(dt)

            self.CheckColorAxis()
            if not self.isCentrale :
                if self.Main.ui.SequenceCheckBox.checkState() == Qt.Checked:
                    self.GlobalData.Slices.PlotOnLastAx()
                self.Main.updateSubPlotFromPlotter(False)
            else :
                self.Main.ui.BufferPlotter.updateSubPlotFromPlotter(False)


            self.canvas.figure.tight_layout() #permet d'ajuster automatiquement les marges si certaines étiquettes sont particulièrement longues (c'est matplotlib qui calcule).
            self.canvas.draw()
            QApplication.restoreOverrideCursor()
        else :
            self.Main.ui.statusbar.showMessage("Nombre de courbe maximale atteinte",SHOW_TIME_NORMAL)
    def delSubPlot(self):

        if len(self.Ax)>1:
            QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
            self.nLigne -= 1
            gs = gridspec.GridSpec(self.nLigne, 1, figure=self.canvas.figure)
            a=self.Ax.pop()
            self.canvas.figure.delaxes( a[0] )
            self.canvas.figure.delaxes( a[1] )
            for i in range(len(self.Ax)):
                self.Ax[i][0].set_subplotspec(gs[i, :])
                self.Ax[i][1].set_subplotspec(gs[i, :])


            self.CheckColorAxis()
            self.toolbar.update()
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            if not self.isCentrale :
                self.Main.updateSubPlotFromPlotter()
            QApplication.restoreOverrideCursor()
    def refreshSubPlot(self):
        self.PlotWhole()
        self.CheckColorAxis()
        self.canvas.draw()

        self.nbOfCursors = 0
        self.listOfCursors.clear()
        self.CursorsIsActive = 0

    def CheckColorAxis(self):
        dt=self.Main.configWidget.timestepComboBox.currentText()
        for i in range(len(self.Ax)):
            self.Ax[i][0].set_xlabel(str(dt))
            if len(self.Ax[i][1].lines) == 0:
                self.Ax[i][1].tick_params(axis='y', labelcolor="white",grid_color='white',color ="white")

            else :
                show=False
                for line in  self.Ax[i][1].lines :
                    if type(line)!=matplotlib.lines._AxLine :
                        show=True
                if show :
                    self.Ax[i][1].tick_params(axis='y', labelcolor="black",grid_color='black',color ="black")
                else :
                    self.Ax[i][1].tick_params(axis='y', labelcolor="white", grid_color='white', color="white")

            if len(self.Ax[i][0].lines) == 0:
                self.Ax[i][0].tick_params(axis='y', labelcolor="white",grid_color='white',color ="white")

            else :
                show = False
                for line in self.Ax[i][0].lines:
                    if type(line) != matplotlib.lines._AxLine:
                        show = True
                if show:
                    self.Ax[i][0].tick_params(axis='y', labelcolor="black",grid_color='black',color ="black")
                else :
                    self.Ax[i][0].tick_params(axis='y', labelcolor="white", grid_color='white', color="white")

    #set global Data variale
    def setGlobalData(self,GD):
        self.GlobalData=GD
    def setMain(self,main,isCentrale=False):
        self.Main=main
        self.isCentrale=isCentrale
    def toogleAxesZOrder(self):
        for twins in self.Ax :
            a=twins[0].get_zorder()
            twins[0].set_zorder( twins[1].get_zorder() )
            twins[1].set_zorder( a)
    def Tranche(self):
            with wait_cursor():
                # should check first if there is already an instance of ListOfSlices
                if len(self.GlobalData.Slices)>0:
                    self.GlobalData.Slices.clear()
                # DV.arValu calculation with selected DV and store it in global Data
                try :
                    start_time = time.time()
                    # seuil sur std des données
                    PSI.PSIDataClass.dDataStdEpsilon =float(self.Main.configWidget.dDataStdEpsilon.text())
                    # seuil sur la pente maxi pour un palier
                    PSI.PSIDataClass.dCoeffAStdEpsilon= float(self.Main.configWidget.dCoeffAStdEpsilon.text())
                    # nb de points minimum dans une séquence
                    PSI.PSIDataClass.Ecart_min= int(self.Main.configWidget.minPoints.text())
                    # nb de points des moyennes mobiles
                    PSI.PSIDataClass.NbPtDetection= int(self.Main.configWidget.NbPtDetection.text())
                    # Oui ou Non: calcul des paliers uniquement
                    PSI.PSIDataClass.bOnlyFlatRangeAreTreated=self.Main.configWidget.comboHandleRamps.currentIndex()
                    # Oui ou non: calcul des paliers avec filtrage des pentes
                    PSI.PSIDataClass.bSlopeIsUsedForFlatRange  =self.Main.configWidget.comboHandleStep_Ramps.currentIndex()
                    # seuil sur la pente maxi pour un palier
                    PSI.PSIDataClass.dCoeffAAvgEpsilon = float(self.Main.configWidget.dCoeffAAvgEpsilon.text())

                    PSI.PSIDataClass.NbPtCoeffADirect=PSI.PSIDataClass.NbPtDetection
                    PSI.PSIDataClass.NbPtAAvg=PSI.PSIDataClass.NbPtDetection
                    # en %, troncature à gauche/droite d'une séquence
                    PSI.PSIDataClass.LeftTronc = int(self.Main.configWidget.LeftTronc.text())
                    PSI.PSIDataClass.RightTronc = int(self.Main.configWidget.RightTronc.text())

                    psi = PSI.PSIDataClass(self.selected_DV.arValu)
                    # check if validated :
                    if psi.Validated !="":
                        self.Main.ErrorMessagePopUp(icon=QMessageBox.Warning,
                            text="Erreur lors du calcul des séquences",info=psi.Validated,
                            title="Erreur de séquençage",buttons=QMessageBox.Cancel)
                    else :
                        #on crée notre variable Slices
                        self.GlobalData.Slices = ListOfSlices(psi.getSliceList(), self.Ax,name=self.selected_DV.stLabl)

                        self.Main.statusBar().showMessage(str(len(self.GlobalData.Slices)) + " séquences calculées ({}s)".format(round(time.time() - start_time,2)),
                                                     SHOW_TIME_NORMAL)
                        #on active l'interraction avec le menu des séquences
                        self.Main.ui.sequenceGroupeBox1.setEnabled(True)
                        self.Main.ui.sequenceGroupeBox2.setEnabled(True)
                        self.Main.refreshSlicesList()
                        self.Main.ui.comboBoxSliceList.setCurrentIndex(self.Main.ui.comboBoxSliceList.findText(self.selected_DV.stLabl))
                        # plotSlices :
                        if self.Main.ui.SequenceCheckBox.checkState() == Qt.Unchecked:
                            self.Main.ui.SequenceCheckBox.setCheckState(Qt.Checked)
                            self.Main.SequenceCheckBoxChange(self.Main.ui.SequenceCheckBox.checkState())
                        else:
                            self.GlobalData.Slices.plot()
                        # update the graph
                        self.canvas.draw()
                except Exception as e:
                    self.Main.ui.statusbar.showMessage("erreur paramètres séquentielles")
                    print(e)
                pass
# --------------------------------------Temps de Réponse Auto (début)-------------------------------------#
    # fonction principale qui gère la procédure
    def tdrAuto(self, data, start, end):
        seuilP = self.Main.configWidget.SeuilPression.text()
        seuilP = float(seuilP)
        detectPression = self.getPressionData(start, end, seuilP)

        # on envoie les données 'data' correspondant aux valeurs dans le zoom
        # à rollingStd pour obtenir l'écart type glissant de la courbe
        std_wlength = int(self.Main.configWidget.Std_wlength.text())
        dataStd = fct_simplified.rollingStd(std_wlength, data)
        # on prend la moyenne glissante
        mg_wlength = int(self.Main.configWidget.MG_wlength.text())
        dataStdMG = fct_simplified.rollingFilter(mg_wlength,0,dataStd)['Values'].tolist()
        # on applique ensuite la normalisation MinMax
        dataStdMGNorm = fct_simplified.MinMaxNormalisation(dataStdMG)

        # ---------- DETERMINATION DES T0 ----------
        dataStdNorm = fct_simplified.MinMaxNormalisation(dataStd)
        seuil = 0.025
        newPression = []
        saveListOfTup = []
        for p in detectPression:
            newPression.append(p)
        newPression.append(1000000)
        # on définit dans une liste de tuples, l'ensemble des points dépassant le seuil
        # on ne détecte les t0 qu'après avoir détecté une variation de pression
        listOfTup = []
        while len(listOfTup) < len(detectPression):
            listOfTup = self.findTupList(seuil, dataStdNorm, newPression)
            seuil = seuil + 0.005
            if len(listOfTup) != len(saveListOfTup):
                length = len(listOfTup) - len(saveListOfTup)
                for i in range(length):
                    saveListOfTup.append(listOfTup[-(i+1)])
                saveListOfTup.sort()
            if seuil > 1:
                #print('seuil limite pour liste de tuple')
                break
        #print('voici la liste de tuples', saveListOfTup)
        # ---------- DETERMINATION DES TF ----------
        # on a nos t0, on vient chercher les tf où le signal se stabilise

        seuil_groupe = 0.0015
        seuil_nbTdr = 0.2
        listOfGroup= []
        saveListOfGroup = []
        saveTypeDeFront = []
        typeDeFront = []
        front_time = []
        nbTdr = 0
        # seuil adaptatif pour la détection du nombre de temps de réponse
        while len(detectPression) != int(nbTdr):
            seuil_nbTdr = seuil_nbTdr - 0.01
            nbTdr, saveTypeDeFront = self.nbTdrAuto(data, seuil_nbTdr, detectPression)
            if seuil_nbTdr < 0.05:
                #print('seuil limite dans l\'estimation du temps de réponse')
                break
        #print('Tuples associés aux fronts détectés: ', saveTypeDeFront)
        for infoFront in saveTypeDeFront:
            coord, front = infoFront
            typeDeFront.append(front)
            front_time.append(coord)

        ''''# on vient filtrer à nouveau les variations de pression
        # on ne garde que celles qui se situent après les front_time
        itemToPop = []
        for i in range(len(detectPression)):
            try:
                ref = front_time[i]
            except:
                break
            if ref > detectPression[i]:
                itemToPop.append(i)
        # for i in itemToPop:
        #    detectPression.pop(i)
        print('MAJ Pression:', detectPression)
        '''
        #print('Type de front détecté:', typeDeFront)
        #print('Nombre de temps de réponse:', nbTdr)

        # seuil adaptatif pour la détection des groupes
        while len(listOfGroup) != int(nbTdr):
            listOfGroup= self.findGroups(seuil_groupe, dataStdMGNorm, saveListOfTup, newPression)
            if len(listOfGroup) != len(saveListOfGroup):
                length = len(listOfGroup) - len(saveListOfGroup)
                for i in range(length):
                    saveListOfGroup.append(listOfGroup[-(i+1)])
                saveListOfGroup.sort()
            seuil_groupe= seuil_groupe + 0.0005
            if seuil_groupe > 1:
                #print('La recherche de groupe n\'a pas aboutit jusqu\'au bout')
                break
        # on vérifie qu'il n'y a pas d'incohérences dans notre liste de groupe
        # c.a.d que les groupes se suivent bien sans être compris l'un dans l'autre
        listOfGroup.append((0,0))
        for i in range(len(saveListOfGroup)):
            t0i, tfi = saveListOfGroup[i]
            t0j, tfj = listOfGroup[i+1]
            if 0 < t0j < tfi:
                saveListOfGroup[i]= listOfGroup[i]
        #print('\nVoici la liste des groupes trouvés:', saveListOfGroup)
        if not saveListOfGroup:
            #print('listOfGroup = listOfTup car empty')
            for tup in saveListOfTup:
                saveListOfGroup.append(tup)

        # ---------- DETERMINATION DES TDR ---------
        # on détermine nos temps de réponse grâce aux groupes créés,
        # aux types de fronts trouvés (montant ou descendant) et nos données
        pourcentage = self.Main.configWidget.SeuilTdr.text()
        listOfTdr = self.CalculTdr(saveListOfGroup, typeDeFront, data, detectPression, int(pourcentage))
        return listOfTdr, saveListOfGroup, typeDeFront

    # crée la liste de tuples grâce à l'écart type glissant normalisé de la courbe
    def findTupList(self, seuil, datastdnorm, pression):
        index = 0
        listOfTup = []
        nbDeTup = 0
        # on définit dans une liste de tuples, l'ensemble des points dépassant le seuil
        # on ne détecte les t0 qu'après avoir détecter une variation de pression
        while index < len(datastdnorm):
            find = False
            if nbDeTup < len(pression) - 1:
                for i in datastdnorm[index:]:
                    if i > seuil and index > pression[nbDeTup]:
                        # t0 = dataStdNorm[index]
                        index_t0 = index
                        find = True
                        break
                    index = index + 1
                if find is True:
                    for f in datastdnorm[index:]:
                        if f < seuil and index < pression[nbDeTup + 1]:
                            # t = dataStdNorm[index]
                            newtup = (index_t0, index)
                            listOfTup.append(newtup)
                            nbDeTup = nbDeTup + 1
                            break
                        index = index + 1
            else:
                index = index + 1
        return listOfTup
    # renvoie les groupes à étudier, trouve les temps où le signal se stabilise
    def findGroups(self, seuil, dataStdMGNorm, listTup, pression):
        index_MG = 0
        listOfGroup = []
        nbDeGroupe = 0
        # on parcourt la moyenne normalisée de l'écart type glissant
        # on s'assure qu'un nouveau groupe ne peut être créé qu'après avoir passé une variation de pression
        for tup in listTup:
            if nbDeGroupe < len(pression) - 1:
                t0, t = tup
                if t0 > index_MG and t0 > pression[nbDeGroupe]:
                    index_MG = t0
                    for i in dataStdMGNorm[t0 + 500:]:
                        if i < seuil and pression[nbDeGroupe + 1] > index_MG > t:
                            tf = index_MG
                            newGroup = (t0, tf)
                            listOfGroup.append(newGroup)
                            nbDeGroupe = nbDeGroupe + 1
                            break
                        index_MG = index_MG + 1
        return listOfGroup

    # récupère les données de PRESSION_AMONT_BAR
    # renvoie une liste avec les temps de variations de pression pour calculer les temps morts
    def getPressionData(self, start, end, seuilP):
        ptFinal = []

        # on vient récupérer le nom de l'étiquette contenant les données de pression,
        # (renseignée par l'utilisateur dans la configuration)
        etiquette = self.Main.configWidget.Pression.text()
        try:
            index = 0
            for item in self.GlobalData.arData:
                if item.stName == etiquette:
                    pression = self.GlobalData.arData[index]
                index = index + 1
            dataP = pression.arValu
        except:
            print('Pas de courbe de pression nommée "PRESSION_AMONT_BAR" détectée')
            return ptFinal
        # crée le même zoom que celui effectué sur nos données
        dataP = dataP[int(start): int(end)]
        # on prend son écart type glissant
        listOfStdDataP= fct_simplified.rollingStd(2000, dataP)
        # on le normalise
        listF = fct_simplified.MinMaxNormalisation(listOfStdDataP)
        #on pose notre seuil et relève les valeurs qui le dépassent
        index = 0
        ptDetection = []
        flag = False
        for x in listF:
            if x > seuilP and flag is False:
                ptDetection.append(index)
                flag = True
            if seuilP > x:
                flag = False
            index = index + 1
        # on a les index en 0,2 maintenant on les veut le plus proche possible de 0 pour être précis
        # difficile d'avoir un seuil_final inférieur à 0,1 car après on ne repasse pas forcément plus bas tout le temps.
        seuil_final = 0.1
        for pt in ptDetection:
            newlistF = listF[:pt]
            index_final = 0
            for x in reversed(newlistF):
                if x < seuil_final:
                    coord_pt = pt - index_final
                    ptFinal.append(coord_pt)
                    break
                index_final = index_final + 1

        # on retire les variations de pression qui arrivent avant X ms
        secToCut = self.Main.configWidget.SecToCut.text()
        for p in ptFinal:
            if p < int(secToCut):
                p_index = ptFinal.index(p)
                ptFinal.pop(p_index)
        return ptFinal

    # renvoie le nombre de temps de réponse à détecter
    def nbTdrAuto(self, data, seuil, pression):
        nbTdr= 0
        # on crée notre liste contenant la moyenne glissante de la dérivée normalisée (n=5000 et w_length=5000)

        dt=self.Main.configWidget.timestepComboBox.currentText()[1:]
        A= fct_simplified.calculDebitDerivation(0, len(data), dt, data)
        A= [x*3600000 for x in A]
        B= fct_simplified.moving_average(A, 6000)
        B= B.tolist()
        B_MinMax = fct_simplified.MinMaxNormalisation(B)
        dataf= fct_simplified.rollingFilter(6000, 0, B_MinMax)

        dataD= dataf['Values'].tolist()

        # on crée nos seuils de détection en fonction de la valeur la plus courante de la liste (valeur stable)
        # on retire les 'nan' de la liste amenées par la dérivée
        newData = [x for x in dataD if math.isnan(x) == False]
        front = []
        moy= mode(newData)
        limMax= moy+seuil
        limMin= moy-seuil
        for x in newData:
            if nbTdr < len(pression):
                if limMax < x and flagM is False and newData.index(x) > pression[nbTdr] - 12000:
                    front.append((newData.index(x),'Montant'))
                    nbTdr = nbTdr + 1
                    flagM = True
                if limMax > x:
                    flagM = False
                if x < limMin and flagD is False and newData.index(x) > pression[nbTdr] - 12000:
                    front.append((newData.index(x),'Descendant'))
                    nbTdr = nbTdr + 1
                    flagD = True
                if x > limMin:
                    flagD = False
        return nbTdr, front

    #calcul des temps de réposne
    def CalculTdr(self, saveListOfGroup, typeDeFront, data, detectPression, pourcentage):
        tdrList = []
        group_index = 0
        # pour protéger si jamais la saveListOfGroup a prit les valeurs de listOfTup
        if len(saveListOfGroup) > len(typeDeFront):
            diff = len(saveListOfGroup) - len(typeDeFront)
            for i in range(diff):
                typeDeFront.append('Erreur')

        for group in saveListOfGroup:
            t0, repere = group
            temps_mort = (t0 - detectPression[group_index]) / 1000
            #print('\n--- ANALYSE DU GROUPE(', t0, repere,')/FRONT', group_index + 1, ' --- temps mort:', temps_mort, 's')
            if typeDeFront[group_index] == 'Montant':
                # on ajoute un léger offset dans la donnée de data[t0] car t0 est souvent qq ms après le front
                #print('f(t0)=', data[t0-50],'f(tf)=', data[repere])
                variation = np.abs(data[repere] - data[t0-50])
                seuil_high = data[repere] + (1-pourcentage/100) * variation
                seuil_low = data[repere] - (1-pourcentage/100) * variation
                #print('variation', variation, 'seuil haut', seuil_high, 'seuil bas', seuil_low)
                seuil_index_up = 0
                # détermination du temps de réponse pour atteindre le seuil de manière stable à x% de Vs
                # on part de tf (donnée stable) et on regarde quand on quitte le seuil de (100-x)%
                newData = data[:repere]
                for value in reversed(newData):
                    if value < seuil_low or value > seuil_high:
                        t_seuil = repere - seuil_index_up
                        tdr_stable = (t_seuil - t0) / 1000 + temps_mort
                        #print('temps de réponse (1) stabilité (%d) à la montée' % pourcentage, tdr_stable, 's')
                        break
                    seuil_index_up = seuil_index_up + 1
                # détermination du tdr dès le moment où on atteint le seuil (même si on le dépasse après)
                # on part de t0 et on regarde dès que l'on dépasse x% de la valeur stable
                index = 0
                for value in data[t0:]:
                    if value > seuil_low:
                        t_pourc = index
                        tdr = t_pourc / 1000 + temps_mort
                        #print('temps de réponse (2) (%d) à la montée' % pourcentage, tdr, 's')
                        break
                    index = index + 1
                tupTdr = (tdr_stable, tdr)
                tdrList.append(tupTdr)
            if typeDeFront[group_index] == 'Descendant':
                #print('f(t0)=', data[t0-50], 'f(tf)=', data[repere])
                variation = np.abs(data[t0-50] - data[repere])
                seuil_high = data[repere] + (1-pourcentage/100) * variation
                seuil_low = data[repere] - (1-pourcentage/100) * variation
                #print('variation', variation, 'seuil haut', seuil_high, 'seuil bas', seuil_low)
                seuil_index_down = 0
                newData = data[:repere]
                # détermination du temps de réponse pour atteindre le seuil de manière stable à x% de Vs
                for value in reversed(newData):
                    if value < seuil_low or value > seuil_high:
                        t_seuil = repere - seuil_index_down
                        tdr_stable = (t_seuil - t0) / 1000 + temps_mort
                        #print('temps de réponse (1) stabilité (%d) à la descente' % pourcentage, tdr_stable, 's')
                        break
                    seuil_index_down = seuil_index_down + 1
                # détermination du tdr dès le moment où on atteint le seuil (même si on le dépasse après)
                index = 0
                for value in data[t0:]:
                    if value < seuil_high:
                        tdr = index / 1000 + temps_mort
                        #print('temps de réponse (2) (%d) à la descente' % pourcentage, tdr, 's')
                        break
                    index = index + 1
                tupTdr = (tdr_stable, tdr)
                tdrList.append(tupTdr)
            if typeDeFront[group_index] == 'Erreur':
                break
            group_index = group_index + 1
        return tdrList
# ---------------------------------------Temps de Réponse Auto (fin)--------------------------------------#
    def cursors(self, y_data, graph_numb, row):
        #for DV in self.GlobalData.arView:
        #    self.selected_DV=DV
        #start, end = x.canvas.axes.get_xlim()
        if y_data != []:
            end = len(y_data) # à changer pour si sur zoom
            if len(y_data) < end:
                end = len(y_data)
            start=0
            yaxis = y_data
            # l'axe des abscisses du curseur dépend de l'axe sur lequel est dessiné la courbe
            if self.Main.ui.subTable.cellWidget(row, 6).currentText() == 'DEFAUT':
                xaxis= np.linspace(start, end, len(yaxis) + 1)
            # si la courbe est dessinée sur un autre axe des abscisses
            else:
                name = self.Main.ui.subTable.item(row, 0).text()
                for DV in self.GlobalData.arView:
                    if DV.stName == name:
                        xaxis = DV.x_value
            if self.CursorsIsActive == 1:
                self.curseur = SnaptoCursor(self.Ax[graph_numb-1][0], xaxis, yaxis)
                cid = self.canvas.mpl_connect('motion_notify_event', self.curseur.mouse_move)
            else:
                self.curseur = SnaptoCursor(self.Ax[graph_numb-1][0], xaxis, yaxis, 1)
        else:
            self.curseur = SnaptoCursor(self.Ax[graph_numb-1][0], [0], [0], 1)
    def addCursors(self, data, graph_numb, row):
        self.nbOfCursors = self.nbOfCursors + 1
        if self.nbOfCursors <= 2:  # on ne peut pas ajouter plus de deux curseurs
            self.CursorsIsActive = 1
            self.cursors(data, graph_numb, row)
        else:
            self.Main.ui.statusbar.showMessage("Nombre de curseurs maximal atteint")
    def delCursors(self):
        self.refreshSubPlot()
    def cursorState(self):
        if self.CursorsIsActive == 0:
            return False
        else:
            return True
    #affiche une nouvelle figure avec le spectre des fréquences de la courbe
    def plotSpectre(self,frq,amplitude):
        plt.figure(figsize=(6,4))
        amp = amplitude.index(max(amplitude))
        frequency = frq[amp]
        '''if frequency != 0:
            print('period', 1/frequency)
        else:
            print('frequence nulle')'''
        frq=np.array(frq)
        amplitude=np.array(amplitude)
        plt.plot(frq, amplitude, label='Spectre de fréquence')
        plt.title('Spectre fréquentielle')
        plt.yscale('log')
        plt.xlabel('fréquence (Hz)')
        plt.ylabel('amplitude')
        plt.legend()
        plt.show()
    def onclick(self,event):
        global x
        x=event
        def crossedStat():
            listOfLines=[]
            linesOnDiffAbs=[]
            start, end = x.canvas.axes.get_xlim()
            save_start, save_end = start, end
            for DV in self.GlobalData.arView :
                if DV.obAxes ==x.inaxes :
                    listOfLines.append(DV.obLine)
                    if DV.x_axis != 'DEFAUT':
                        linesOnDiffAbs.append(DV.obLine)
            # grâce à linesOnDiffAbs on peut traiter séparemment les courbes qui sont sur d'autres abscisses
            # listOfLines=x.inaxes.lines
            dt = TIMESTEP_ENUM[self.Main.configWidget.timestepComboBox.currentText()]
            listOfData=[]
            temp_dataFrame=pd.DataFrame()
            # ici on vient modifier le end de chaque courbe s'il est > à sa taille de donnée
            listOfEnd = []
            for line in listOfLines:
                start, end = save_start, save_end
                if line in linesOnDiffAbs:
                    x_data = line.get_xdata()
                    # on tri notre axe des abscisses (comme sur le graphe)
                    x_data.sort()
                    # on récupère les index de start et end
                    n = 0
                    for data in x_data:
                        if data >= end:
                            pos_end = n
                            break
                        n = n + 1
                    end = pos_end
                    if len(line.get_ydata()) < end:
                        end = len(line.get_ydata())
                    listOfEnd.append(end)
                else:
                    if len(line.get_ydata()) < end:
                        end = len(line.get_ydata())
                    listOfEnd.append(end)
            i = 0
            for line in listOfLines:
                end = listOfEnd[i]
                start = save_start
                if line in linesOnDiffAbs:
                    x_data = line.get_xdata()
                    x_data.sort()
                    m = 0
                    # ici on adapte le start pour le ramener à la bonne échelle
                    for data in x_data:
                        if data >= start:
                            pos_start = m
                            break
                        m = m + 1
                    start = pos_start
                    listOfData.append(line.get_ydata()[int(start):int(end)])
                    try:
                        temp_dataFrame[line.get_label()] = pd.Series(line.get_ydata()[int(start):int(end)])
                    except Exception as e:
                        print(e)
                else:
                    listOfData.append(line.get_ydata()[int(start):int(end)])
                    temp_dataFrame[line.get_label()] = line.get_ydata()[int(start):int(end)]
                i = i + 1
            data = pd.DataFrame()
            for i in range(len(temp_dataFrame.columns)):
                col1=temp_dataFrame.iloc[:, i]
                mean_col1=col1.mean()
                std_col1=col1.std()
                name1=temp_dataFrame.columns[i]
                for j in range(i,len(temp_dataFrame.columns)):
                    if i!=j:
                        col2 =temp_dataFrame.iloc[:, j]
                        mean_col2=col2.mean()
                        std_col2=col2.std()
                        name2=temp_dataFrame.columns[j]
                        diff=col1.subtract(col2)

                        if DEBUG :
                            plt.plot(diff)
                            plt.show()
                        described_diff=diff.describe()
                        moyenne_relative= pd.Series([(mean_col2-mean_col1)/mean_col1 *100, (mean_col1-mean_col2)/mean_col2 *100], index=['Erreur relative (B-A)/A %', 'Erreur relative (A-B)/B %'])
                        ecart_type_relatif=pd.Series([-(std_col1-std_col2)/mean_col1 *100, -(std_col2-std_col1)/mean_col2 *100], index=['Ecart-type relatif (std(B)-std(A))/moy(A) %',
                                                                                                                                        'Ecart-type relatif (std(A)-std(B))/moy(B) %'])
                        relativ_stability=pd.Series([std_col2/std_col1*100,std_col1/std_col2*100],index=["Stabilité relative A %","Stabilité relative B %"])

                        described_diff=described_diff.append(moyenne_relative).append(ecart_type_relatif).append(relativ_stability)
                        described_diff=described_diff.round(DATAFRAME_PRECISION).rename({'count': 'Nombre de points',
                                    'mean': 'Valeur moyenne','std': 'Ecart-type','min': 'Valeur mini','max': 'Valeur maxi'})
                        data["A : "+name1+" - B : "+name2]=described_diff


            self.InfoTable.emit(data)
        def placeCursors():
            self.CursorsIsActive= 0
            self.listOfCursors.append(self.curseur.xcoords)
            self.listOfCursors.append(self.curseur.ycoords)
            # le row ici n'a pas d'intérêt
            self.cursors([], 1,row= 0)
        def acquireCursors():
            step = TIMESTEP_ENUM[self.Main.configWidget.timestepComboBox.currentText()]
            freq = self.Main.configWidget.Frequence.text()
            freq = float(freq)
            dt, dt_step, dy, dephasage = self.curseur.measurement(self.listOfCursors, step, freq)
            self.Main.cursorResults(dt, dt_step, dy, dephasage)

        if event.button== 2:
            self.toolbar.release_pan(x)
            self.toolbar.release_zoom(x)
            self.popMenu2 = QMenu(self)
            self.popMenu2.setStyleSheet("color: black;font-weight:bold;")


            if self.nbOfCursors>=1 and self.CursorsIsActive==1:
                noteAction = QAction(QIcon(""), "Placer un curseur", self, triggered=placeCursors)
                self.popMenu2.addAction(noteAction)

            if self.nbOfCursors == 2 and self.CursorsIsActive==0:
                noteAction = QAction(QIcon(""), "Faire une mesure", self, triggered=acquireCursors)
                self.popMenu2.addAction(noteAction)

            cursor = QCursor()
            self.popMenu2.popup(cursor.pos())
            self.Picked_Flag = False
        if event.button== 3:
            self.toolbar.release_pan(x)
            self.toolbar.release_zoom(x)
            self.popMenu = QMenu(self)
            self.popMenu.setStyleSheet("color: black;font-weight:bold;")

            noteAction= QAction(QIcon(""), "Choix de l'axe "+("Droit" if self.Ax[0][0].get_zorder()==1 else "Gauche")+" pour les actions futures de la figure",self,
                                       triggered = self.toogleAxesZOrder)
            self.popMenu.addAction(noteAction)

            noteAction= QAction(QIcon(""), "Détermination des statistiques croisées des courbes de la figure",self,
                                           triggered = crossedStat)
            self.popMenu.addAction(noteAction)

            if self.Picked_Flag:
                noteAction = QAction(QIcon(""), "Définition des séquences à partir de la variable "+self.selected_DV.stName, self,
                                     triggered=self.Tranche)
                self.popMenu.addAction(noteAction)

            cursor = QCursor()
            self.popMenu.popup(cursor.pos())
            self.Picked_Flag = False

    def onpick(self,event):
        if event.mouseevent.button == 3:
            for DV in self.GlobalData.arView:
                if event.artist == DV.obLine:
                    self.selected_DV=DV
                    self.Picked_Flag=True

        if event.mouseevent.button == 1:
            if event.artist.get_label() != '':
                self.annot.remove()
                annot = event.artist.axes.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                                               bbox=dict(boxstyle="round", fc="w"),
                                                               arrowprops=dict(arrowstyle="->"))
                text=event.artist.get_label()+"\n"
                start, end = event.canvas.axes.get_xlim()
                zoomed =event.artist.get_ydata()[int(start) if int(start) >=0 else 0:int(end)]

                data = pd.DataFrame(zoomed,
                                    index=[str(i) for i in range(len(zoomed))]).describe().round(DATAFRAME_PRECISION).rename(index={'count': 'Nombre de points',
                                    'mean': 'Valeur moyenne','std': 'Ecart-type','min': 'Valeur mini','max': 'Valeur maxi'})
                # remove the quartiles
                data.drop('25%', inplace=True, axis=0)
                data.drop('50%', inplace=True, axis=0)
                data.drop('75%', inplace=True, axis=0)
                zoomed_pd=pd.Series(zoomed)
                relativ_stability = pd.DataFrame([round(zoomed_pd.std() / zoomed_pd.mean(),DATAFRAME_PRECISION)],
                                              index=["Ecart-type relatif"])

                data = data.append(relativ_stability)

                text+=str(data).split("\n",1)[1]

                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(1)
                annot.xy = (event.mouseevent.xdata,event.mouseevent.ydata)
                annot.set_visible(True)
                self.annot=annot
                self.canvas.draw()
            else:
                self.annot.set_visible(False)
                self.canvas.draw()

    #drag and drop event
    def dragEnterEvent(self, e) :
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
    def dragMoveEvent(self,e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
    def dropEvent(self, e):
        if e.mimeData().hasUrls and not self.isCentrale:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()
            for url in e.mimeData().urls():
                fname = str(url.toLocalFile())
                break
            self.Main.OpenFileRequest(drop=True,fileName=fname)
        else:
            e.ignore()  # just like above functions

    def update_Timer(self, refreshPeriod):
        self.timer.stop()
        self.timer.start(refreshPeriod)

    def PlotWhole(self):
        self.timer.stop()
        # clear all axes, then plot all dataVari in arView
        [x[0].clear() for x in self.Ax]
        for x in self.Ax :
            x[1].clear()
            x[1].yaxis.set_label_position("right")
            x[1].yaxis.tick_right()
        # plot every DV in arView

        [x.plot() for x in self.GlobalData.arView]
        self.ptr=0
        self.canvas.draw()

    def PlotOnNewAbs(self, ax, index):
        # on est obligé de clear le graphe sinon l'ancienne figure reste
        ax.clear()
        # ici on replot tout
        [x.plot() for x in self.GlobalData.arView]
        #self.GlobalData.arView[index].plot(Ax=ax,x_value=xdata)
        self.canvas.draw()

    #step function
    def update(self):
        #si on a des courbes sélectionnées
        for DV in self.GlobalData.arView:
            StopUpdate=DV.selectivePlot(self.ptr,self.ptr + self.interval)
        self.ptr+=self.step
        dt=self.Main.configWidget.timestepComboBox.currentText()[1:]
        for x in self.Ax :
            x[0].relim()
            x[0].autoscale_view()
            x[1].relim()
            x[1].autoscale_view()
            x[0].tick_params(bottom=True, top=False, left=True, right=False)
            x[0].set_xlabel(dt)

        if StopUpdate :
            self.PlotWhole()
        else :
            self.canvas.draw()

    def ClearALl(self):
        # clear all axes
        for x in self.Ax:
            x[0].clear()
            x[0].tick_params(bottom=True, top=False, left=True, right=False)
            x[0].tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)

            x[1].clear()
            #x[1].tick_params(bottom=True, top=False, left=False, right=True)
            x[1].tick_params(labelbottom=True,labeltop=False,labelleft=False,labelright=True)
            x[1].tick_params(bottom=True, top=False, left=False, right=True)
            #x[1].yaxis.set_label_position("right")
            #x[1].yaxis.tick_right()
        self.canvas.draw()







