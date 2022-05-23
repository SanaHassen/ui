
import cv2
from label import *
from mplwidget import *
import imutils
import os, glob
import numpy as np
import math, time
import queue
import threading

class ProcessingUnitThreads:
    def __init__(self, initial_frame, input_data, result ) -> None:
        self.initial_frame = initial_frame
        self.input_data = input_data
        self.result = result
        self.output_queue = queue.Queue()
        self.input_queue = queue.Queue()


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
        positions = [0]
        threads = []
        all_images = self.load_images()
        images_filenames = all_images[1:]
        time_instants = self.get_time_from_frame_rate(all_images) 
        images_filenames.sort()
        
        for i in range(10):
            worker = threading.Thread(target=self.threads_job, args=(), daemon=True)
            worker.start()
            threads.append(worker)

        for image_path in images_filenames:
            self.input_queue.put(image_path)

        # put this in another thread
        while len(positions) <= len(images_filenames): # len(positions) = len(images_filenames) + 1
            positions.append(self.output_queue.get())
            self.output_queue.task_done()
        
        self.input_queue.join()
        self.output_queue.join()

        return positions,time_instants
        

    def threads_job(self):
        while True: 
            image_path = self.input_queue.get()
            try:
                img=cv2.imread(image_path)
                img = img[self.initial_frame.ROI_LeftCorner_y:self.initial_frame.ROI_RightCorner_y, self.initial_frame.ROI_LeftCorner_x:self.initial_frame.ROI_RightCorner_x]
            except:
                self.set_notification("problème de chargement de l'image ")
                return

            try:
                if self.initial_frame.modulo  != 0: 
                    img=imutils.rotate_bound(img, 90*self.initial_frame.modulo)
            except:
                self.set_notification("problème dans la rotation des images ")
                return
    
            correlation_ROI_template = cv2.matchTemplate(img,self.initial_frame.template_frame,cv2.TM_SQDIFF)
            min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(correlation_ROI_template)
            correlation_ROI_template = correlation_ROI_template[0,:]
            x = np.arange(len(correlation_ROI_template))
            min_loc = min_loc[0]
            fit_params=np.polyfit(x[min_loc-30:min_loc+30],correlation_ROI_template[min_loc-30:min_loc+30],50) #polyfit: x, y, deg:50 
            x_fit = np.arange(min_loc-30,min_loc+30-1,0.0001)
            y_fit=np.polyval(fit_params, x_fit)
            y_fit=np.asarray(y_fit)
            min_fit=min(y_fit) 
            min_loc_fit=x_fit[np.where(y_fit==min_fit)[0]][0]

            self.output_queue.put(min_loc_fit)
            self.input_queue.task_done()




    def calculate_flow_velocity(self):
        self.result.positions, self.result.time_instants = self.position_per_time_calculation()
        print(self.result.positions, len(self.result.positions))
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
  