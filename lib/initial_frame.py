from PyQt5.QtWidgets import QFileDialog
import cv2
from label import *
from mplwidget import *
import imutils



class InitialFrame: 
    def __init__(self) -> None:

        # Initial file name  
        self.file_name = ''

        # rotation
        self.counter = 0
        self.modulo = 0

        # original_frame
        self.original_frame = None
        self.original_frame_width = 0
        self.original_frame_height = 0

        # Template frame 
        self.template_frame = None
        self.template_width = None
        self.template_height = None
        self.template_LeftCorner_x = None
        self.template_LeftCorner_y = 0
        self.template_RightCorner_x = 0
        self.template_RightCorner_y = 0

        # Roi Frame
        self.ROI_frame = None
        self.ROI_width = 0
        self.ROI_height = 0   
        self.ROI_LeftCorner_x = 0
        self.ROI_LeftCorner_y = None
        self.ROI_RightCorner_x = None
        self.ROI_RightCorner_y =  None
    
    def import_initial_frame(self) : 
        self.file_name, _ = QFileDialog.getOpenFileName(None,"Importation des données image",filter="Images(*.tif)")
        self.original_frame = cv2.imread(self.file_name)
        self.original_frame_height,self.original_frame_width,ch = self.original_frame.shape
        print("original image shape: ", self.original_frame_width,self.original_frame_height)
        self.ROI_frame = self.original_frame.copy()
        self.ROI_width = self.ROI_frame.shape[1]
        self.ROI_height = self.ROI_frame.shape[0]
        print("ROI", self.ROI_frame.shape)

    def rotate_frame(self):
        self.ROI_frame=imutils.rotate_bound(self.ROI_frame, 90)
        self.counter += 1
        self.modulo=self.counter%4
    
    def apply_roi_selection(self,x0,y0,x1,y1, max_width, max_height): 
        self.ROI_LeftCorner_x = int((x0*self.ROI_frame.shape[1])/max_width)
        self.ROI_LeftCorner_y = int((y0*self.ROI_frame.shape[0])/max_height)
        if (x1 < max_width):
            self.ROI_RightCorner_x = int((x1*self.ROI_frame.shape[1])/max_width)
        else:
            self.ROI_RightCorner_x = self.original_frame_width
        
        if(y1 < max_height):
            self.ROI_RightCorner_y =  int((y1*self.ROI_frame.shape[0])/max_height)
        else:
            self.ROI_RightCorner_y = self.original_frame_height

        print('ROI coordinates',self.ROI_LeftCorner_x,self.ROI_LeftCorner_y,self.ROI_RightCorner_x, self.ROI_RightCorner_y)
        self.ROI_frame = self.ROI_frame[self.ROI_LeftCorner_y:self.ROI_RightCorner_y,self.ROI_LeftCorner_x:self.ROI_RightCorner_x]
        self.ROI_width = self.ROI_frame.shape[1]
        self.ROI_height = self.ROI_frame.shape[0]

    def apply_template_selection(self, x0,y0,x1,y1,x0_test,y0_test, max_width, max_height):
        if x0 != x0_test:
            x0 = x0_test
            y0 = y0_test
        self.template_LeftCorner_x = int((x0*self.ROI_frame.shape[1])/max_width)
        self.template_LeftCorner_y = 0
        self.template_RightCorner_x = int((x1*self.ROI_frame.shape[1])/max_width)
        self.template_RightCorner_y = self.ROI_height
        self.template_width = abs(self.template_RightCorner_x - self.template_LeftCorner_x)
        self.template_height = abs(self.template_RightCorner_y - self.template_LeftCorner_y)
        print('left',self.template_LeftCorner_x,self.template_LeftCorner_y,self.template_RightCorner_x, self.template_RightCorner_y)
        self.template_frame = self.ROI_frame[self.template_LeftCorner_y:self.template_RightCorner_y,self.template_LeftCorner_x:self.template_RightCorner_x]

    def validate(self): 
        if  self.ROI_frame is None:
            raise Exception("Veuillez sélectionner une image")
        elif self.template_LeftCorner_x is None:
            raise Exception("Veuillez sélectionner un template et/ou ROI")
