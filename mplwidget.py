from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib
import matplotlib.pyplot as plt


# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')

# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.fig, (self.ax1, self.ax2,self.ax3) = plt.subplots(3, 1)
        self.ax1.set_title('Interface Positions vs. Time')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Position (pixels)')
        self.ax2.set_title('Dynamic Flow Rate vs. Time')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Dynamic flow rate (µl/min)')
        self.ax3.set_title('Distribution of Dynamic Flow rate')
        self.ax3.set_xlabel('Dynamic Flow rate (µl/min)')
        self.ax3.set_ylabel('Count')

        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.4, 
                            hspace=0.5)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

# Matplotlib widget
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()                  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)