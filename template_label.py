from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter,QPen
from PyQt5.QtCore import QRect, Qt

class Template_Label(QLabel):
 x0 = 0
 y0 = 0
 x1 = 0
 y1 = 0
 flag = False
 count = 0
 #callback = None
 #Mouse click event
 def mousePressEvent(self,event):
  self.flag = True
  self.x0 = event.x()
  self.y0 = event.y()
 #Mouse release event
 def mouseReleaseEvent(self,event):
  self.flag = False
  #if self.callback:
  self.callback(self.x0,self.y0,self.x1,self.y1,self.x0_test,self.y0_test)
  #self.callback = None
 #Mouse movement events
 def mouseMoveEvent(self,event):
  if self.flag:
   self.x1 = event.x()
   self.y1 = event.y()
   self.x0_test = self.x0
   self.y0_test = self.y0
   self.update()
 #Draw events
 def paintEvent(self, event):
  super().paintEvent(event)
  #if not self.callback: 
  #  return
  rect =QRect(self.x0, self.y0, abs(self.x1-self.x0), abs(self.y1-self.y0))
  #self.count += 1
  painter = QPainter(self)
  painter.setPen(QPen(Qt.red,2,Qt.SolidLine))
  painter.drawRect(rect)