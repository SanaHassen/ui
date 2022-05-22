from PyQt5 import uic
import sys
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtWidgets import QFileDialog
import cv2
from PyQt5.QtGui import QPixmap,QImage,QFont
from PyQt5.QtCore import Qt
from label import *
from mplwidget import *
import imutils
import os, glob
import numpy as np
import pandas as pd
import math, time
#from sklearn.linear_model import LinearRegression



class Recap(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Recap, self).__init__(parent)
        uic.loadUi('recap.ui', self)
        self.title.setFont(QFont("Times", 10, QFont.Bold))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('nano_debit_interface_2eme_version.ui', self)
        self.image_holder.setGeometry(11,372,1024,416)
        self.counter = 0
        self.modulo = 0
        self.original_image_width = 0
        self.original_image_height = 0
        self.original_image = None
        self.ROI_image = None
        self.template_image = None
        self.taillePixel = 0
        self.frame_rate = 0
        self.new_frame_rate = 0
        self.diametre = 0
        self.timestamp = 0
        self.fname = ''
        self.IsNewFrequence = False
        self.using_timestamp = False
        self.ROI_width = 0
        self.ROI_height = 0
        self.template_width = None
        self.template_height = None
        self.ROI_LeftCorner_x = 0
        self.ROI_LeftCorner_y = None
        self.ROI_RightCorner_x = None
        self.ROI_RightCorner_y =  None
        self.template_LeftCorner_x = 0
        self.template_LeftCorner_y = 0
        self.template_RightCorner_x = 0
        self.template_RightCorner_y = 0
        self.velocity = 0
        self.residuals = 0
        self.positions = []
        self.time_instants = []
        self.flow_rate_value = 0.0
        self.start = 0
        self.head = ''
        self.slope = 0 
        self.intercept = 0
        self.predicted_positions = []
        self.flow_rate_dynm = 0
        self.splitter.setStretchFactor(1, 7)
           

        self.bouton_select.clicked.connect(self.on_selection_button_clicked)
        self.bouton_rot.clicked.connect(self.on_rotation_button_clicked)
        self.bouton_roi.clicked.connect(self.on_ROI_button_clicked)
        self.reset_button.clicked.connect(self.on_reset_button_clicked)
        self.bouton_template.clicked.connect(self.on_template_button_clicked)
        self.bouton_select.setToolTip(" Cliquer sur ce bouton pour selectionner une image du local")
        self.bouton_roi.setToolTip(" Pour séléctionner une région d'interet de l'image, Veuillez tout d'abord: \n 1-faire bouger le curseur sur l'image \n 2-dessiner un rectangle sur l'image \n 3- lachez")
        self.bouton_template.setToolTip(" Refetes le meme process que selection de ROI \n Si tout est bien passé vous trouverez une notification tout en bas indiquant que la template est selectionnée et sauvegardée")
        self.boutton_mesure_debit.clicked.connect(self.on_mesure_debit_button_clicked)
        self.check_nouvelle_freq.stateChanged.connect(self.checkNouvelleFreqChangedAction)
        self.timestamp_checkbox.stateChanged.connect(self.checkTimeStampsChangedAction)
        #self.taille_pixel.editingFinished.connect(self.enter_taille_pixel)
        #self.freq_acquisition.editingFinished.connect(self.enter_freq_acquisition)
        #self.nouvelle_freq.editingFinished.connect(self.enter_nouvelle_freq)
        #self.diameter.editingFinished.connect(self.enter_diameter)
        #self.timestamps_start.editingFinished.connect(self.enter_timestamps_start)
        self.bouton_enregistre_debit.clicked.connect(self.on_enregiste_debit_button_clicked)
        self.recap = Recap(self)
        self.recap.confirmer.clicked.connect(self.enregistrer_recap) # => bouton "Ok"
        self.recap.button_annuler.clicked.connect(self.annuler_recap) # => bouton "Annuler"
        # self.taillePixel = self.taille_pixel.text()
        # self.frame_rate = self.freq_acquisition.text()
        # self.new_frame_rate = self.nouvelle_freq.text()
        # self.diametre = self.diameter.text()
        # self.timestamp = self.timestamps_start.text()


    def update_view(self,image,placement): # roi
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, ch = rgb_image.shape
            bytes_per_line = ch * image_width
            convert_to_Qt_format = QImage(rgb_image.data, image_width, image_height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_Qt_format)
            placement.setPixmap(pixmap.scaled(self.image_holder.width(),self.image_holder.height(),Qt.KeepAspectRatio))

    def update_view_2(self,image,placement): # template
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, ch = rgb_image.shape
        bytes_per_line = ch * image_width
        convert_to_Qt_format = QImage(rgb_image.data, image_width, image_height, bytes_per_line, QImage.Format_RGB888)
        pixmap_image = QPixmap.fromImage(convert_to_Qt_format)
        # create painter instance with pixmap
        painterInstance = QPainter(pixmap_image)
        # set rectangle color and thickness
        penRectangle = QPen(QtCore.Qt.red)
        penRectangle.setWidth(3)
        # draw rectangle on painter
        painterInstance.setPen(penRectangle)
        painterInstance.drawRect(self.template_LeftCorner_x, self.template_LeftCorner_y,self.template_width, self.template_height)       
        # set pixmap onto the label widget
        placement.setPixmap(pixmap_image)
        painterInstance.end()
                           
    def on_selection_button_clicked(self):
        try:  
            self.notifications.setText('')
            self.fname, _ = QFileDialog.getOpenFileName(None,"Importation des données image",filter="Images(*.tif)")
            self.original_image = cv2.imread(self.fname)
            self.original_image_height,self.original_image_width,ch = self.original_image.shape
            print(self.original_image_width,self.original_image_height)
            self.ROI_image = self.original_image.copy()
            print("ROI", self.ROI_image.shape)
            self.update_view(self.ROI_image,self.image_holder)
        except : 
            self.notifications.setText('Veuillez sélectionner une image')
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))
    
    def images_are_equals(self,image1,image2):
        if image2 is None:
            self.notifications.setText('Veuillez sélectionner une image')
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))
            return
        elif image1.shape == image2.shape:
            return True
        else:
            return False
         
    def on_rotation_button_clicked(self):
        try:
            self.notifications.setText('')
            self.ROI_image=imutils.rotate_bound(self.ROI_image, 90)
            self.update_view(self.ROI_image,self.image_holder)
            self.counter += 1
            self.modulo=self.counter%4
        except:
            self.notifications.setText('Veuillez sélectionner une image')
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))
            
    def on_ROI_button_clicked(self):
        try:
            self.notifications.setText('')
            self.image_holder.setCursor(Qt.CrossCursor)
            self.image_holder.callback = self.on_roi_selection_complete
        except:
            self.notifications.setText('Veuillez sélectionner une image')
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))

    def on_reset_button_clicked(self):
        
        if self.ROI_image is None:
            self.notifications.setText('Veuillez sélectionner une image')
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))
        else:
            self.notifications.setText('')
            self.ROI_image = self.original_image
            self.update_view(self.ROI_image,self.image_holder)
            

    def on_template_button_clicked(self):
        try:
            self.notifications.setText('')
            self.image_holder.setCursor(Qt.CrossCursor)
            self.image_holder.callback = self.on_template_selection_complete
        except:
            self.notifications.setText('Veuillez sélectionner une image')
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))
        
    def on_roi_selection_complete(self,x0,y0,x1,y1,x0_test,y0_test):
        try:
            self.ROI_LeftCorner_x = int((x0*self.ROI_image.shape[1])/self.image_holder.width())
            self.ROI_LeftCorner_y = int((y0*self.ROI_image.shape[0])/self.image_holder.height())
            if (x1 < self.image_holder.width()):
                self.ROI_RightCorner_x = int((x1*self.ROI_image.shape[1])/self.image_holder.width())
            else:
                self.ROI_RightCorner_x = self.original_image_width
            
            if(y1 < self.image_holder.height()):
                self.ROI_RightCorner_y =  int((y1*self.ROI_image.shape[0])/self.image_holder.height())
            else:
                self.ROI_RightCorner_y = self.original_image_height

            print('left',self.ROI_LeftCorner_x,self.ROI_LeftCorner_y,self.ROI_RightCorner_x, self.ROI_RightCorner_y)
            self.ROI_image = self.ROI_image[self.ROI_LeftCorner_y:self.ROI_RightCorner_y,self.ROI_LeftCorner_x:self.ROI_RightCorner_x]
            self.ROI_width = self.ROI_image.shape[1]
            self.ROI_height = self.ROI_image.shape[0]
            self.update_view(self.ROI_image,self.image_holder)
        except:
            self.notifications.setText('Veuillez sélectionner une image')
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))

    def on_template_selection_complete(self,x0,y0,x1,y1,x0_test,y0_test):
        if x0 != x0_test:
            x0 = x0_test
            y0 = y0_test
        self.template_LeftCorner_x = int((x0*self.ROI_image.shape[1])/self.image_holder.width())
        self.template_LeftCorner_y = 0
        self.template_RightCorner_x = int((x1*self.ROI_image.shape[1])/self.image_holder.width())
        self.template_RightCorner_y = self.ROI_height
        self.template_width = abs(self.template_RightCorner_x - self.template_LeftCorner_x)
        self.template_height = abs(self.template_RightCorner_y - self.template_LeftCorner_y)
        print('left',self.template_LeftCorner_x,self.template_LeftCorner_y,self.template_RightCorner_x, self.template_RightCorner_y)
        self.template_image = self.ROI_image[self.template_LeftCorner_y:self.template_RightCorner_y,self.template_LeftCorner_x:self.template_RightCorner_x]
        self.update_view_2(self.ROI_image,self.image_holder)
        self.notifications.setText("template selectionnéee et sauvergardée")
        self.notifications.setFont(QFont("Times", 10, QFont.Bold))
       

    def checkNouvelleFreqChangedAction(self,state):
        if (QtCore.Qt.Checked == state):
            self.nouvelle_freq_label.setEnabled(True)
            self.nouvelle_freq.setEnabled(True)
            self.IsNewFrequence = True
        else:
            self.nouvelle_freq_label.setEnabled(False)
            self.nouvelle_freq.setEnabled(False)

    def checkTimeStampsChangedAction(self,state):
        if (QtCore.Qt.Checked == state):
            self.timestamp_label.setEnabled(True)
            self.bouton_select_temps.setEnabled(True)
            self.label_timestamp.setEnabled(True)
            self.timestamps_start.setEnabled(True)
            self.using_timestamp = True

        else:
            self.timestamp_label.setEnabled(False)
            self.bouton_select_temps.setEnabled(False)
            self.label_timestamp.setEnabled(False)
            self.timestamps_start.setEnabled(False)

    def images_storage(self,images_path):
        images_new_filenames = []
        self.head = os.path.split(images_path)[0]
        images_filenames = glob.glob(self.head + "/*.tif")
        try:
            if (self.IsNewFrequence == False or not self.new_frame_rate): 
                return images_filenames
            else:
                for i in range(0,len(images_filenames)):
                    j=i*int(self.frame_rate/self.new_frame_rate)
                    if j<len(images_filenames):
                        images_new_filenames.append(images_filenames[j])
                    else:
                        break
                return images_new_filenames
        except:
            self.notifications.setText("Un problème est survenu lors de l'importation des images")
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))
    
    def time_from_frame_rate(self,images_filenames):
            time_instants=[]
            for i in range (0,len(images_filenames)):
                if (self.IsNewFrequence == False or not self.new_frame_rate):
                    time_instants.append(i/self.frame_rate)
                else:
                    time_instants.append(i/self.new_frame_rate)
            return time_instants

    def position_per_time_calculation(self):
        pos = 0
        counter=0
        displacements,template_xLeft_coordinates,ROI_xLeft_coordinates,positions = [],[],[],[]
        n=0
        images_filenames = self.images_storage(self.fname)
        time_instants = self.time_from_frame_rate(images_filenames)
        new_template_LeftCorner_x = self.template_LeftCorner_x
        images_filenames.sort()
        images_equals = self.images_are_equals(self.original_image,self.ROI_image)
        for i in range(0,len(images_filenames)-1):
            print(images_filenames[i])
            try:
                img1=cv2.imread(images_filenames[i])
                img2=cv2.imread(images_filenames[i+1])  
            except:
                self.notifications.setText("problème de chargement de l'image " + str(i))
                self.notifications.setFont(QFont("Times", 10, QFont.Bold))
                return

            try:
                img1=imutils.rotate_bound(img1, 90*self.modulo)
                #cv2.imshow("a",img1)
                img2=imutils.rotate_bound(img2, 90*self.modulo)
                #cv2.imshow("b",img2)

            except:
                self.notifications.setText("problème dans la rotation des images ")
                self.notifications.setFont(QFont("Times", 10, QFont.Bold))
                return
            # print("ROI_LeftCorner_x",self.ROI_LeftCorner_x)
            # print("ROI_width",self.ROI_width)
            # print("calcul", self.ROI_LeftCorner_x + self.ROI_width + np.sum(displacements))
            # print("width",self.original_image_width)
            if images_equals == False:
                counter += 1
                if (self.ROI_LeftCorner_x + self.ROI_width + int(np.sum(displacements)) < self.original_image_width):  # quand ROI n'a pas atteint le bord de l'image
                    case = "case 1"
                    self.ROI_LeftCorner_x_new=int (self.ROI_LeftCorner_x+ np.sum(displacements))
                    print("left_x", self.ROI_LeftCorner_x, "left_new",self.ROI_LeftCorner_x_new)
                    self.ROI_RightCorner_x_new =int (self.ROI_LeftCorner_x+self.ROI_width + np.sum(displacements))
                    print("right_x", self.ROI_RightCorner_x, "right_new",self.ROI_RightCorner_x_new)
                    img1=img1[self.ROI_LeftCorner_y:self.ROI_RightCorner_y, self.ROI_LeftCorner_x_new:self.ROI_RightCorner_x_new]
                    img2=img2[self.ROI_LeftCorner_y:self.ROI_RightCorner_y, self.ROI_LeftCorner_x_new:self.ROI_RightCorner_x_new]
                    #Define the template from img1, keeping same intial coordinates (see explanation above)
                    img_template=img1[self.template_LeftCorner_y:self.template_RightCorner_y,self.template_LeftCorner_x:self.template_RightCorner_x]
                elif (self.ROI_LeftCorner_x + self.ROI_width + int(np.sum(displacements)) >= self.original_image_width):  # onfixe le ROI  et on change les coordonnées de la template
                    case = "case 2"
                    self.ROI_LeftCorner_x_new =self.original_image_width - self.ROI_width
                    self.ROI_RightCorner_x_new = self.original_image_width
                    img1=img1[self.ROI_LeftCorner_y:self.ROI_RightCorner_y, self.ROI_LeftCorner_x_new:self.ROI_RightCorner_x_new]
                    img2=img2[self.ROI_LeftCorner_y:self.ROI_RightCorner_y, self.ROI_LeftCorner_x_new:self.ROI_RightCorner_x_new]
                    #Define the template from img1,The ROI being fixed the template must keep moving with the interface
                    new_template_LeftCorner_x=self.template_LeftCorner_x+int(np.sum(displacements[i-1-n:i-1]))
                    new_template_RightCorner_x=self.template_RightCorner_x+int(np.sum(displacements[i-1-n:i-1]))
                    img_template=img1[self.template_LeftCorner_y:self.template_RightCorner_y,new_template_LeftCorner_x:new_template_RightCorner_x]
                    n=n+1
                    if new_template_RightCorner_x >= self.ROI_width-50:
                        break

            else:   
                    case = "case 3"
                    counter=counter+1
                    new_template_LeftCorner_x=self.template_LeftCorner_x+int(np.sum(displacements))
                    new_template_RightCorner_x=self.template_RightCorner_x+int(np.sum(displacements))
                    img_template=img1[self.template_LeftCorner_y:self.template_RightCorner_y,new_template_LeftCorner_x:new_template_RightCorner_x]
                    n=n+1
                    if new_template_RightCorner_x >= self.original_image_width - 50:
                        break 

            #template_xLeft_coordinates.append(new_template_LeftCorner_x)
            #ROI_xLeft_coordinates.append(self.ROI_LeftCorner_x_new)
           
            # try:        
            #     img_template = cv2.GaussianBlur(img_template,(5,5),cv2.BORDER_DEFAULT)
            #     img2 = cv2.GaussianBlur(img2,(5,5),cv2.BORDER_DEFAULT)

            # except:
                # self.notifications.setText("ERREUR!! le template est vide")
                # self.notifications.setFont(QFont("Times", 10, QFont.Bold))
                # return
            
            correlation_ROI_template = cv2.matchTemplate(img2,img_template,cv2.TM_SQDIFF)
            min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(correlation_ROI_template)
            correlation_ROI_template = correlation_ROI_template[0,:]
            #corr_array = np.asarray(correlation_ROI_template[0,:])
            #print("len_corr_aaray",len(corr_array))
            x = np.arange(len(correlation_ROI_template))
            #min_cor=min(corr_array)
            #min_loc=x[np.where(corr_array==min_cor)[0]][0] 
            min_loc = min_loc[0]
            fit_params=np.polyfit(x[min_loc-30:min_loc+30],correlation_ROI_template[min_loc-30:min_loc+30],50) #polyfit: x, y, deg:50 
            #create a polymnome using fit_params with x_fit : x abs of 0.0001 resolution and y_fit from x_fit and fit_params
            x_fit = np.arange(min_loc-30,min_loc+30-1,0.0001) 
            y_fit=np.polyval(fit_params, x_fit)
            y_fit=np.asarray(y_fit)
            #find min after 0.0001 pixel resolution
            min_fit=min(y_fit)
            min_loc_fit=x_fit[np.where(y_fit==min_fit)[0]][0]
            #Bmin_val,max_val,min_loc,max_loc = cv2.minMaxLoc(y_fit)
            displacements.append(min_loc_fit-new_template_LeftCorner_x)

        positions.append(pos)
        print(counter)
        for j in range (0,counter):
            pos=pos+displacements[j] 
            positions.append(pos)
    
        return positions,time_instants
    
    def flow_velocity(self):
        self.positions, self.time_instants = self.position_per_time_calculation()
        #Linear regression of positions vs. time ( A is the slope == Velocity)
        reg_coeffs,residuals,_,_,_=np.polyfit(self.time_instants,self.positions,1,full=True)
        self.A,self.B=reg_coeffs
        time_array=np.array(self.time_instants)   
        self.predicted_positions = self.A*time_array + self.B
        return self.A*self.taillePixel, residuals[0]*(self.taillePixel)**2
        


        # x = np.array(self.positions).reshape((-1, 1))
        # print('x', x.shape)
        # y = np.array(self.time_instants)
        # print('y',y.shape)
        # model = LinearRegression()
        # model.fit(x, y)
        # self.intercept = model.intercept_
        # self.slope = model.coef_
        # self.predicted_positions = model.predict(x)
        # residuals = (y - self.predicted_positions)
        # velocity = self.slope*self.taillePixel
        # error = residuals*(self.taillePixel)**2
        # return velocity, error

    def Dynamic_flow_rate(self): # calculer la vitesse entre deux images successives
        vitesse_dynamique=[]
        flow_rate_dynamique=[]
        #positions, time_instants = self.position_per_time_calculation()
        for i in range (0,len(self.positions)-2):
            vitesse_dynamique.append((self.positions[i+1]-self.positions[i])/(self.time_instants[i+1]-self.time_instants[i])*self.taillePixel)
        vitesse_dynamique=np.asarray(vitesse_dynamique)
        flow_rate_dynamique=vitesse_dynamique*math.pi*(float(self.diametre)/2)**2*60*1e-09
        return flow_rate_dynamique
    
    def on_mesure_debit_button_clicked(self):
        #evaluate duration of flow rate calculation
        if  self.ROI_image is None:
            self.notifications.setText("Veuillez sélectionner une image")
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))
            return
        elif self.template_LeftCorner_x is None:
            self.notifications.setText("Veuillez sélectionner une template et/ou ROI")
            self.notifications.setFont(QFont("Times", 10, QFont.Bold))
            return

        self.start = time.process_time()
        
        try:
             self.taillePixel = float(self.taille_pixel.text())
             self.frame_rate = float(self.freq_acquisition.text())
             self.diametre = float(self.diameter.text())
        except:
             self.notifications.setText("un ou plusieurs paramètres sont manquants ou invalides")
             self.notifications.setFont(QFont("Times", 10, QFont.Bold))
             return
           
        if self.IsNewFrequence == True:
            try:
                 self.new_frame_rate = float(self.nouvelle_freq.text())
            except ZeroDivisionError:
             self.notifications.setText("un ou plusieurs paramètres sont manquants ou invalides")
             self.notifications.setFont(QFont("Times", 10, QFont.Bold))

        if self.using_timestamp == True:
            try:
                self.timestamp = float(self.timestamps_start.text())
            except:
                self.notifications.setText("un ou plusieurs paramètres sont manquants ou invalides")
                self.notifications.setFont(QFont("Times", 10, QFont.Bold))

        self.velocity,self.residuals=self.flow_velocity()
        print("Process time=", time.process_time() - self.start)

        self.flow_rate_value=self.velocity*math.pi*(float(self.diametre)/2)**2*60*1e-09
        self.flow_rate.setText(str(self.flow_rate_value))
        self.notifications.setText("débit calculé")
        self.notifications.setFont(QFont("Times", 10, QFont.Bold))
        self.dynm_flow_rate = self.Dynamic_flow_rate()

    def calculate_parameters(self):
        self.positions = np.asarray(self.positions)
        self.time_instants = np.asarray(self.time_instants)
        self.mean_position=np.mean(self.positions)
        self.positions_nomber= len(self.positions)
        self.time_instants_number=len(self.time_instants)
        self.time_instants_dynm=self.time_instants[1:len(self.time_instants)-1]

    def on_enregiste_debit_button_clicked(self):
        self.calculate_parameters()
        self.recap.taille_pixel.setText(str(self.taillePixel))
        self.recap.diametre.setText(str(self.diametre))
        self.recap.frate_rame.setText(str(self.frame_rate))
        self.recap.flow_velocity.setText(str(self.velocity))
        self.recap.flow.setText(str(self.flow_rate_value))
        self.recap.mean_position.setText(str(self.mean_position))
        self.recap.nb_position.setText(str(self.positions_nomber))
        self.recap.nb_instants.setText(str(self.time_instants_number))
        self.recap.process_time.setText(str(time.process_time() - self.start))
        self.recap.show()

    def enregistrer_recap(self):
            #save individual parameters
            sys.stdout = open('Results.txt', "+w")
            print("Taille de pixel (micro-meter)=", self.taillePixel)
            print("Diametre interne du capillaire (micro-meter)=", self.diametre)
            print("images path=",self.head)
            print("Frequence d'acquisition=", self.frame_rate, " , Nouvelle frequence d'acquisition=",self.new_frame_rate)
            print("Flow velocity (micro-meter/s)=", self.velocity)
            print("flow rate (micro-meter/min)=",self.flow_rate_value)
            print("Residuals(Pixels²)=",self.residuals)
            print("Mean position (pixels) =",self.mean_position)
            print("Degrees of freedom=",len(self.positions)-2)
            print("len positions=",len(self.positions),"len time instants=",len(self.time_instants))
            #print("Relative Residual Standard Deviation/error=",math.sqrt((self.residuals)/(len(self.positions)-2))/self.mean_position)
            print("Process time=", time.process_time() - self.start)
            sys.stdout.close()
            self.recap.close() 

            #save postion_time_data
            output = {"time_instants": self.time_instants,"positions":self.positions}
            output_data = pd.DataFrame(data=output)
            with open("positions_times_data.txt", '+w') as f:
                dfAsString = output_data.to_string(index=False)
                f.write(dfAsString)
                

            #save ROI and Template cooridnates
            info_ROI_template_file="ROI_Temp_infos.txt"
            coordinate_ROI_Temp_file="ROI_Temp_coordinates.txt"
            
            output_data = pd.DataFrame({"Roi width":[self.ROI_width] ,"Roi Height":[self.ROI_height],"Template width":[self.template_width]})
            with open(info_ROI_template_file, '+w') as f:
                f.write(" ---------------------- fichier contenant des infos sur la ROI et le template ---------------------- ")
                dfAsString = output_data.to_string(index=False)
                f.write(dfAsString)
           

            ROI_coordinates = [self.ROI_LeftCorner_x,self.ROI_LeftCorner_y,self.ROI_RightCorner_x, self.ROI_RightCorner_y]
            template_coordinates = [self.template_LeftCorner_x,self.template_LeftCorner_y,self.template_RightCorner_x, self.template_RightCorner_y]
            index = ["x_left","y_left","x_right","y_right"]
            output = {"location":index,"ROI_coordinates":ROI_coordinates ,"template_coordinates": template_coordinates}
            output_data = pd.DataFrame(data=output)
            output_data.set_index("location")
            with open(coordinate_ROI_Temp_file, '+w') as f:
                f.write(" ---------------------- fichier contenant les coordonnées de ROI et de template ----------------------  ")
                dfAsString = output_data.to_string()
                f.write(dfAsString)
            self.tabs.setCurrentIndex(1)
            # x=range(0, 10)
            # y=range(0, 20, 2)
            # self.plot_data.canvas.ax1.scatter(x,y,s=10,c='crimson',marker='x', label='Measured positions')
            # self.plot_data.canvas.ax1.plot(x, y, '-r', c='dodgerblue', label='Linear fit')
            # self.plot_data.canvas.ax2.scatter(x,y,s=10,c='crimson',marker='x')
            # self.plot_data.canvas.ax2.plot(x,y,'x-', color="crimson")
            # self.plot_data.canvas.ax3.hist(x, bins='auto',color='black',edgecolor='black')
            # self.plot_data.canvas.draw()

            #Uself.tabs.setCurrentIndex(1)
            self.plot_data.canvas.ax1.scatter(self.time_instants,self.positions,s=10,c='crimson',marker='x', label='Measured positions')
            self.plot_data.canvas.ax1.plot(self.time_instants,self.predicted_positions, '-r', c='dodgerblue', label='Linear fit')
            self.plot_data.canvas.ax2.plot(self.time_instants_dynm,self.dynm_flow_rate,'x-', color="crimson")
            self.plot_data.canvas.ax3.hist(self.dynm_flow_rate, bins='auto',color='dodgerblue',edgecolor='black')
            self.plot_data.canvas.draw()

           

    
    def annuler_recap(self):
            self.recap.close()
        



def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()