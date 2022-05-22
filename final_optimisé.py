# select template directly : wrong behaviour 

from PyQt5 import uic
import sys
from PyQt5 import QtCore,QtWidgets
import cv2
from PyQt5.QtGui import QPixmap,QImage,QFont
from PyQt5.QtCore import Qt
from label import *
from mplwidget import *
import numpy as np
import pandas as pd
import  time
from lib import InitialFrame, Result, ProcessingUnit, InputData, ProcessingUnitThreads


class SummaryWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(SummaryWindow, self).__init__(parent)
        uic.loadUi('summary_dialog.ui', self)
        self.title.setFont(QFont("Times", 10, QFont.Bold))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('nano_debit_interface_2eme_version_refactored.ui', self)


        self.initial_frame = InitialFrame()
        self.input_data = InputData()
        self.result = Result()
        
        #self.processing_unit = ProcessingUnitThreads(self.initial_frame,self.input_data,self.result)
        self.processing_unit = ProcessingUnit(self.initial_frame,self.input_data,self.result)

        self.ui_image_holder.setGeometry(11,372,1024,416)

        # ui_splitter 
        self.ui_splitter.setStretchFactor(1, 7)
           
        # ui_select_start_image_button
        self.ui_select_start_image_button.clicked.connect(self.on_select_image_button_clicked)
        self.ui_select_start_image_button.setToolTip(" Cliquer sur ce bouton pour selectionner une image du local")

        # ui_rotate_image_button
        self.ui_rotate_image_button.clicked.connect(self.on_rotate_image_button_clicked)

        # ui_select_roi_button
        self.ui_select_roi_button.clicked.connect(self.on_select_roi_button_clicked)
        self.ui_select_roi_button.setToolTip(" Pour séléctionner une région d'interet de l'image, Veuillez tout d'abord: \n 1-faire bouger le curseur sur l'image \n 2-dessiner un rectangle sur l'image \n 3- lachez")

        # ui_reset_roi_button 
        self.ui_reset_roi_button.clicked.connect(self.on_reset_roi_button_clicked)

        # ui_select_template_button 
        self.ui_select_template_button.clicked.connect(self.on_select_template_button_clicked)
        self.ui_select_template_button.setToolTip(" Refetes le meme process que selection de ROI \n Si tout est bien passé vous trouverez une notification tout en bas indiquant que la template est selectionnée et sauvegardée")
        
        # ui_calculate_flow_rate_button 
        self.ui_calculate_flow_rate_button.clicked.connect(self.on_calculate_flow_rate_button_clicked)

        # ui_new_frequency_checkbox
        self.ui_new_frequency_checkbox.stateChanged.connect(self.on_new_frequency_checkbox_changed)

        # ui_allow_timestamp_checkbox
        self.ui_allow_timestamp_checkbox.stateChanged.connect(self.on_allow_timestamp_checkbox_changed)

        # ui_save_data_button
        self.ui_save_data_button.clicked.connect(self.on_save_data_button_clicked)

        # ui_notifications
        self.ui_notifications.setFont(QFont("Times", 10, QFont.Bold))


        self.summary_dialog = SummaryWindow(self)
        
        # summary_confirm_button
        self.summary_dialog.summary_confirm_button.clicked.connect(self.on_summary_confirm_button_clicked) 

        # summary_cancel_button
        self.summary_dialog.summary_cancel_button.clicked.connect(self.on_summary_cancel_button_clicked) 

    def update_ui_image_holder(self,image,placement): 
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, ch = rgb_image.shape
            bytes_per_line = ch * image_width
            convert_to_Qt_format = QImage(rgb_image.data, image_width, image_height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_Qt_format)
            placement.setPixmap(pixmap.scaled(self.ui_image_holder.width(),self.ui_image_holder.height(),Qt.KeepAspectRatio))

    def update_ui_image_holder_with_template(self,image,placement): 
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, ch = rgb_image.shape
        bytes_per_line = ch * image_width
        convert_to_Qt_format = QImage(rgb_image.data, image_width, image_height, bytes_per_line, QImage.Format_RGB888)
        pixmap_image = QPixmap.fromImage(convert_to_Qt_format)
        painterInstance = QPainter(pixmap_image)
        penRectangle = QPen(QtCore.Qt.red)
        penRectangle.setWidth(3)
        painterInstance.setPen(penRectangle)
        painterInstance.drawRect(self.initial_frame.template_LeftCorner_x, self.initial_frame.template_LeftCorner_y,self.initial_frame.template_width, self.initial_frame.template_height)       
        placement.setPixmap(pixmap_image)
        painterInstance.end()


    def reset_notification(self): 
        self.set_notification('')
        self.ui_notifications.setStyleSheet('color: black')


    def set_notification(self, message, type='INFO'): 
        self.ui_notifications.setText(message)
        if type == 'ERROR': 
            self.ui_notifications.setStyleSheet('color: red')
        elif type == 'SUCCESS':
            self.ui_notifications.setStyleSheet('color: green')
        else:
            self.ui_notifications.setStyleSheet('color: black')


    # Button Actions                       
    def on_select_image_button_clicked(self):
        try:  
            self.set_notification('')
            self.initial_frame.import_initial_frame()
            self.update_ui_image_holder(self.initial_frame.ROI_frame,self.ui_image_holder)
        except : 
            self.set_notification('Veuillez sélectionner une image')
            
    
    def on_rotate_image_button_clicked(self):
        try:
            self.set_notification('')
            self.initial_frame.rotate_frame()
            self.update_ui_image_holder(self.initial_frame.ROI_frame,self.ui_image_holder)
        except:
            self.set_notification('Veuillez sélectionner une image')
            
            
    def on_select_roi_button_clicked(self):
        try:
            self.set_notification('')
            self.ui_image_holder.setCursor(Qt.CrossCursor)
            self.ui_image_holder.callback = self.on_roi_selection_complete
        except:
            self.set_notification('Veuillez sélectionner une image')
            

    def on_reset_roi_button_clicked(self):
        if self.initial_frame.ROI_frame is None:
            self.set_notification('Veuillez sélectionner une image')
        else:
            self.set_notification('')
            self.initial_frame.ROI_frame = self.initial_frame.original_frame
            self.update_ui_image_holder(self.initial_frame.ROI_frame,self.ui_image_holder)
            

    def on_select_template_button_clicked(self):
        try:
            self.set_notification('')
            self.ui_image_holder.setCursor(Qt.CrossCursor)
            self.ui_image_holder.callback = self.on_template_selection_complete
        except:
            self.set_notification('Veuillez sélectionner une image')
            
        
    def on_roi_selection_complete(self,x0,y0,x1,y1,x0_test,y0_test):
        try:
            self.initial_frame.apply_roi_selection(x0=x0,y0=y0,x1=x1,y1=y1,max_width=self.ui_image_holder.width(), max_height=self.ui_image_holder.height())
            self.update_ui_image_holder(self.initial_frame.ROI_frame,self.ui_image_holder)
        except:
            self.set_notification('Veuillez sélectionner une image')
            

    def on_template_selection_complete(self,x0,y0,x1,y1,x0_test,y0_test):
        self.initial_frame.apply_template_selection(x0,y0,x1,y1,x0_test,y0_test,max_width=self.ui_image_holder.width(), max_height=self.ui_image_holder.height())
        self.update_ui_image_holder_with_template(self.initial_frame.ROI_frame,self.ui_image_holder)
        self.set_notification("template selectionnéee et sauvergardée")
        

    def on_new_frequency_checkbox_changed(self,state):
        is_checked = QtCore.Qt.Checked == state

        self.nouvelle_freq_label.setEnabled(is_checked)
        self.nouvelle_freq.setEnabled(is_checked)
        self.input_data.IsNewFrequence = is_checked

    def on_allow_timestamp_checkbox_changed(self,state):
        is_checked = QtCore.Qt.Checked == state

        self.timestamp_label.setEnabled(is_checked)
        self.bouton_select_temps.setEnabled(is_checked)
        self.label_timestamp.setEnabled(is_checked)
        self.timestamps_start.setEnabled(is_checked)
        self.input_data.using_timestamp = is_checked

   
    def validate_input_data(self): 
        try:
            self.input_data.taillePixel = float(self.taille_pixel.text()) #test
            self.input_data.frame_rate = float(self.freq_acquisition.text())
            self.input_data.diametre = float(self.diameter.text())

            # if true, pass sinn raise error
            assert  self.input_data.frame_rate != 0

            if self.input_data.IsNewFrequence == True:    
                    self.input_data.new_frame_rate = float(self.nouvelle_freq.text())  
                    assert self.input_data.new_frame_rate != 0
                    
            if self.input_data.using_timestamp == True:
                    self.input_data.timestamp = float(self.timestamps_start.text())
            
        except:
            raise Exception("un ou plusieurs paramètres sont manquants ou invalides")   

    def on_calculate_flow_rate_button_clicked(self):  
        
        try:
            self.initial_frame.validate()
            self.validate_input_data()
        except BaseException as err:
            self.set_notification(str(err), 'ERROR')
            return    

        start_time = time.process_time()

        self.set_notification("Calculating..")
        self.result.start = start_time

        self.processing_unit.calculate_flow_velocity()
        self.processing_unit.calculate_flow_rate()
        self.processing_unit.calculate_dynamic_flow_rate()



        end_time = time.process_time()
        print("Process time=",  start_time, end_time)
        print("Process time=", end_time - start_time)

        self.ui_flow_rate_result.setText(str(self.result.flow_rate_value))
        self.set_notification("débit calculé", 'SUCCESS')


    def calculate_parameters(self):
        self.result.positions = np.asarray(self.result.positions)
        self.result.time_instants = np.asarray(self.result.time_instants)
        self.result.mean_position=np.mean(self.result.positions)
        self.result.positions_nomber= len(self.result.positions)
        self.result.time_instants_number=len(self.result.time_instants)
        self.result.time_instants_dynm=self.result.time_instants[1:len(self.result.time_instants)-1]

    def on_save_data_button_clicked(self):
        self.calculate_parameters()
        self.summary_dialog.taille_pixel.setText(str(self.input_data.taillePixel))
        self.summary_dialog.diametre.setText(str(self.input_data.diametre))
        self.summary_dialog.frate_rame.setText(str(self.input_data.frame_rate))
        self.summary_dialog.flow_velocity.setText(str(self.result.velocity))
        self.summary_dialog.flow.setText(str(self.result.flow_rate_value))
        self.summary_dialog.mean_position.setText(str(self.result.mean_position))
        self.summary_dialog.nb_position.setText(str(self.result.positions_nomber))
        self.summary_dialog.nb_instants.setText(str(self.result.time_instants_number))
        self.summary_dialog.process_time.setText(str(time.process_time() - self.result.start))
        self.summary_dialog.show()

    def on_summary_confirm_button_clicked(self):
            info_ROI_template_file="ROI_Temp_infos.txt"
            coordinate_ROI_Temp_file="ROI_Temp_coordinates.txt"
            result_filename= "Results.txt"

            sys.stdout = open(result_filename, "+w")
            print("Taille de pixel (micro-meter)=", self.input_data.taillePixel)
            print("Diametre interne du capillaire (micro-meter)=", self.input_data.diametre)
            print("images path=",self.result.head)
            print("Frequence d'acquisition=", self.input_data.frame_rate, " , Nouvelle frequence d'acquisition=",self.input_data.new_frame_rate)
            print("Flow velocity (micro-meter/s)=", self.result.velocity)
            print("flow rate (micro-meter/min)=",self.result.flow_rate_value)
            print("Residuals(Pixels²)=",self.result.residuals)
            print("Mean position (pixels) =",self.result.mean_position)
            print("Degrees of freedom=",len(self.result.positions)-2)
            print("len positions=",len(self.result.positions),"len time instants=",len(self.result.time_instants))
            #print("Relative Residual Standard Deviation/error=",math.sqrt((self.result.residuals)/(len(self.result.positions)-2))/self.result.mean_position)
            print("Process time=", time.process_time() - self.result.start)
            sys.stdout.close()
            self.summary_dialog.close() 

            output = {"time_instants": self.result.time_instants,"positions":self.result.positions}
            output_data = pd.DataFrame(data=output)
            with open("positions_times_data.txt", '+w') as f:
                dfAsString = output_data.to_string(index=False)
                f.write(dfAsString)
                


            
            output_data = pd.DataFrame({"Roi width":[self.initial_frame.ROI_width] ,"Roi Height":[self.initial_frame.ROI_height],"Template width":[self.initial_frame.template_width]})
            with open(info_ROI_template_file, '+w') as f:
                f.write(" ---------------------- fichier contenant des infos sur la ROI et le template ---------------------- ")
                dfAsString = output_data.to_string(index=False)
                f.write(dfAsString)
           

            ROI_coordinates = [self.initial_frame.ROI_LeftCorner_x,self.initial_frame.ROI_LeftCorner_y,self.initial_frame.ROI_RightCorner_x, self.initial_frame.ROI_RightCorner_y]
            template_coordinates = [self.initial_frame.template_LeftCorner_x,self.initial_frame.template_LeftCorner_y,self.initial_frame.template_RightCorner_x, self.initial_frame.template_RightCorner_y]
            index = ["x_left","y_left","x_right","y_right"]
            output = {"location":index,"ROI_coordinates":ROI_coordinates ,"template_coordinates": template_coordinates}
            output_data = pd.DataFrame(data=output)
            output_data.set_index("location")
            with open(coordinate_ROI_Temp_file, '+w') as f:
                f.write(" ---------------------- fichier contenant les coordonnées de ROI et de template ----------------------  ")
                dfAsString = output_data.to_string()
                f.write(dfAsString)
            self.tabs.setCurrentIndex(1)
            self.plot_data.canvas.ax1.scatter(self.result.time_instants,self.result.positions,s=10,c='crimson',marker='x', label='Measured positions')
            self.plot_data.canvas.ax1.plot(self.result.time_instants,self.result.predicted_positions, '-r', c='dodgerblue', label='Linear fit')
            self.plot_data.canvas.ax2.plot(self.result.time_instants_dynm,self.result.flow_rate_dynm,'x-', color="crimson")
            self.plot_data.canvas.ax3.hist(self.result.flow_rate_dynm, bins='auto',color='dodgerblue',edgecolor='black')
            self.plot_data.canvas.draw()

    def on_summary_cancel_button_clicked(self):
            self.summary_dialog.close()
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()