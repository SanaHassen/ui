
import cv2
from label import *
from mplwidget import *
import imutils
import os, glob
import numpy as np
import math, time
import sys

class ProcessingUnit:
    def __init__(self, initial_frame, input_data, result ) -> None:
        self.initial_frame = initial_frame
        self.input_data = input_data
        self.result = result

    def are_images_equals(self,image1,image2):
        return image1.shape == image2.shape

    def load_images(self):
        images_new_filenames = []
        self.result.head = os.path.split(self.initial_frame.file_name)[0]
        images_filenames = glob.glob(self.result.head + "/*.tif")
        if (self.input_data.IsNewFrequence == False or not self.input_data.new_frame_rate): 
            return images_filenames
        else:
            for i in range(0,len(images_filenames)):
                j=i*int(self.input_data.frame_rate/self.input_data.new_frame_rate)
                if j<len(images_filenames):
                    images_new_filenames.append(images_filenames[j])
                else:
                    break
            return images_new_filenames 

    def get_time_from_frame_rate(self,images_filenames):
            time_instants=[]
            for i in range (0,len(images_filenames)):
                if (self.input_data.IsNewFrequence == False or not self.input_data.new_frame_rate):
                    time_instants.append(i/self.input_data.frame_rate)
                else:
                    time_instants.append(i/self.input_data.new_frame_rate)
            return time_instants

    def position_per_time_calculation(self):
        pos = 0
        counter=0
        displacements = []
        positions = []
        n=0
        start_time_begin = time.perf_counter()
        images_filenames = self.load_images() 
        load_images_time =  time.perf_counter() - start_time_begin
        time_instants = self.get_time_from_frame_rate(images_filenames) 

        new_template_LeftCorner_x = self.initial_frame.template_LeftCorner_x
        images_filenames.sort()

        images_equals = self.are_images_equals(self.initial_frame.original_frame,self.initial_frame.ROI_frame)
        
        for i in range(0,len(images_filenames)-1):
            current_displacement = int(np.sum(displacements))
            if i == 0:
               
                try:
                    img1=cv2.imread(images_filenames[i])
                    img2=cv2.imread(images_filenames[i+1])
                    
                except:
                    #self.set_notification("problème de chargement de l'image " + str(i))
                    return

                try:
                    if self.initial_frame.modulo  != 0: 
                        img1=imutils.rotate_bound(img1, 90*self.initial_frame.modulo)
                        img2=imutils.rotate_bound(img2, 90*self.initial_frame.modulo)
                except:
                    #self.set_notification("problème dans la rotation des images ")
                    return
            else:
                try:
                    img1 = img2_copy
                    img2 = cv2.imread(images_filenames[i+1])
                except:
                    #self.set_notification("problème de chargement de l'image " + str(i))
                    return

                try:
                    if self.initial_frame.modulo  != 0: 
                        img2=imutils.rotate_bound(img2, 90*self.initial_frame.modulo)
                except:
                    #self.set_notification("problème dans la rotation des images ")
                    return

            img2_copy = img2.copy()

            if images_equals == False:
                counter += 1
                if (self.initial_frame.ROI_LeftCorner_x + self.initial_frame.ROI_width + current_displacement < self.initial_frame.original_frame_width):  # quand ROI n'a pas atteint le bord de l'image
                    self.initial_frame.ROI_LeftCorner_x_new=int (self.initial_frame.ROI_LeftCorner_x+ np.sum(displacements))
                    
                    self.initial_frame.ROI_RightCorner_x_new =int (self.initial_frame.ROI_LeftCorner_x+self.initial_frame.ROI_width + np.sum(displacements))
                   
                    img1=img1[self.initial_frame.ROI_LeftCorner_y:self.initial_frame.ROI_RightCorner_y, self.initial_frame.ROI_LeftCorner_x_new:self.initial_frame.ROI_RightCorner_x_new]
                    img2=img2[self.initial_frame.ROI_LeftCorner_y:self.initial_frame.ROI_RightCorner_y, self.initial_frame.ROI_LeftCorner_x_new:self.initial_frame.ROI_RightCorner_x_new]
                    img_template=img1[self.initial_frame.template_LeftCorner_y:self.initial_frame.template_RightCorner_y,self.initial_frame.template_LeftCorner_x:self.initial_frame.template_RightCorner_x]
               
                elif (self.initial_frame.ROI_LeftCorner_x + self.initial_frame.ROI_width + current_displacement >= self.initial_frame.original_frame_width):  # onfixe le ROI  et on change les coordonnées de la template
                   
                    self.initial_frame.ROI_LeftCorner_x_new =self.initial_frame.original_frame_width - self.initial_frame.ROI_width
                    self.initial_frame.ROI_RightCorner_x_new = self.initial_frame.original_frame_width
                    img1=img1[self.initial_frame.ROI_LeftCorner_y:self.initial_frame.ROI_RightCorner_y, self.initial_frame.ROI_LeftCorner_x_new:self.initial_frame.ROI_RightCorner_x_new]
                    img2=img2[self.initial_frame.ROI_LeftCorner_y:self.initial_frame.ROI_RightCorner_y, self.initial_frame.ROI_LeftCorner_x_new:self.initial_frame.ROI_RightCorner_x_new]
                    new_template_LeftCorner_x=self.initial_frame.template_LeftCorner_x+int(np.sum(displacements[i-1-n:i-1]))
                    new_template_RightCorner_x=self.initial_frame.template_RightCorner_x+int(np.sum(displacements[i-1-n:i-1]))
                    img_template=img1[self.initial_frame.template_LeftCorner_y:self.initial_frame.template_RightCorner_y,new_template_LeftCorner_x:new_template_RightCorner_x]
                    n=n+1
                    if new_template_RightCorner_x >= self.initial_frame.ROI_width-50:
                        break

            else:   
                    counter=counter+1
                    new_template_LeftCorner_x=self.initial_frame.template_LeftCorner_x+current_displacement
                    new_template_RightCorner_x=self.initial_frame.template_RightCorner_x+current_displacement
                    img_template=img1[self.initial_frame.template_LeftCorner_y:self.initial_frame.template_RightCorner_y,new_template_LeftCorner_x:new_template_RightCorner_x]
                    n=n+1
                    if new_template_RightCorner_x >= self.initial_frame.original_frame_width - 50:
                        break 
           
            correlation_ROI_template = cv2.matchTemplate(img2,img_template,cv2.TM_SQDIFF)
            min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(correlation_ROI_template)
            correlation_ROI_template = correlation_ROI_template[0,:]
            x = np.arange(len(correlation_ROI_template))
            min_loc = min_loc[0]
            try:
                fit_params=np.polyfit(x[min_loc-30:min_loc+30],correlation_ROI_template[min_loc-30:min_loc+30],10)
            except:
                return 
            x_fit = np.arange(min_loc-30,min_loc+30-1,0.01)
            y_fit=np.polyval(fit_params, x_fit)
            y_fit=np.asarray(y_fit)
            min_fit=min(y_fit)
            min_loc_fit=x_fit[np.where(y_fit==min_fit)[0]][0]
            displacements.append(min_loc_fit-new_template_LeftCorner_x)
            print(f"processing images {i} and {i+1} ....")
        

        positions.append(pos)
        for j in range (0,counter):
            pos=pos+displacements[j] 
            positions.append(pos)
        
        whole_process_time =   time.perf_counter() - start_time_begin
        print("whole process time", whole_process_time)

        return positions,time_instants

    def calculate_flow_velocity(self):
        self.result.positions, self.result.time_instants = self.position_per_time_calculation()

        reg_coeffs,residuals,_,_,_=np.polyfit(self.result.time_instants,self.result.positions,1,full=True)
        self.A,self.B=reg_coeffs
        time_array=np.array(self.result.time_instants)   
        self.result.predicted_positions = self.A*time_array + self.B
        self.result.velocity = self.A*self.input_data.taillePixel
        self.result.residuals = residuals*(self.input_data.taillePixel)**2


    def calculate_flow_rate(self) : 
        self.result.flow_rate_value=self.result.velocity*math.pi*(float(self.input_data.diametre)/2)**2*60*1e-09


    def calculate_dynamic_flow_rate(self): 
        vitesse_dynamique=[]
        flow_rate_dynamique=[]
        for i in range (0,len(self.result.positions)-2):
            vitesse_dynamique.append((self.result.positions[i+1]-self.result.positions[i])/(self.result.time_instants[i+1]-self.result.time_instants[i])*self.input_data.taillePixel)
        vitesse_dynamique=np.asarray(vitesse_dynamique)
        flow_rate_dynamique=vitesse_dynamique*math.pi*(float(self.input_data.diametre)/2)**2*60*1e-09
        self.result.flow_rate_dynm = flow_rate_dynamique

   
  