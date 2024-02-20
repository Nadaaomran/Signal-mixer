from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QPen
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QRect, QPoint, QSize
from scipy.fft import fft2, fftshift
from PIL import Image
import math
from PyQt5 import QtCore, QtGui, QtWidgets

class Image:
    def __init__(self):
        self.mag=[]
        self.phase=[]
        self.real=[]
        self.imaginary=[]
        self.pix_mag= None
        self.pix_phase=None
        self.pix_real= None
        self.pix_imaginary=None
        self.freq=None
        self.img=None
        self.file_path= None
        self.fourier= None

    def fourier_transform(self, minimum_width, minimum_height):
        image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)   
        if image.shape[1] != minimum_width or image.shape[0]!= minimum_height :
            image = cv2.resize(image, (minimum_width,minimum_height))
        fourier = np.fft.fft2(image) 
        # Shift the zero-frequency component to the center of the spectrum
        self.fourier = np.fft.fftshift(fourier)
    
    def FT_magnitude(self):
        # calculate the magnitude of the Fourier Transform
        self.mag= np.abs(self.fourier)

    def set_pix_magnitude(self):   
        # Scale the magnitude for display
        magnitude = cv2.normalize(20*np.log(self.mag), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # Convert magnitude array to QImage
        magnitude_image = QImage(
            magnitude.data,
            magnitude.shape[1],
            magnitude.shape[0],
            magnitude.shape[1],
            QImage.Format_Grayscale8,
        )

        # Create a QPixmap from the QImage
        self.pix_mag = QPixmap(magnitude_image)
    
    def FT_phase(self):
        # calculate the phase of the Fourier Transform
        self.phase = np.angle(self.fourier)

    def set_pix_phase(self):
        # Scale the phase for display
        phase = cv2.normalize(self.phase, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # Convert phaes array to QImage
        phase_image = QImage(
            phase.data,
            phase.shape[1],
            phase.shape[0],
            phase.shape[1],
            QImage.Format_Grayscale8,
        )
        
        # Create a QPixmap from the QImage
        self.pix_phase = QPixmap.fromImage(phase_image)
    
    def FT_real(self):
        self.real = np.real(self.fourier)

    def set_pix_real(self):
        # Scale the real part for display
        real_part = cv2.normalize(self.real, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # Convert real part array to QImage
        real_image = QImage(
            real_part.data,
            real_part.shape[1],
            real_part.shape[0],
            real_part.shape[1],
            QImage.Format_Grayscale8,
        )

        # Create a QPixmap from the QImage
        self.pix_real = QPixmap.fromImage(real_image)
    
    def FT_imaginary(self):
        self.imaginary = np.imag(self.fourier) 
        
    def set_pix_imaginary(self):
        # Scale the imaginary part for display
        imaginary_part = cv2.normalize(self.imaginary, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # Convert magnitude, phaes, real part and imaginary part arrays to QImage
        imaginary_image = QImage(
            imaginary_part.data,
            imaginary_part.shape[1],
            imaginary_part.shape[0],
            imaginary_part.shape[1],
            QImage.Format_Grayscale8,
        )

        # Create a QPixmap from the QImage
        self.pix_imaginary = QPixmap.fromImage(imaginary_image)
    
    def get_image_attributes(self, size, minimum_width, minimum_height):
        self.create_cutomized_image( size, minimum_width, minimum_height)
        self.fourier_transform(minimum_width, minimum_height)
        self.get_output_related_attributes()
        self.get_pix_maps()

    def get_output_related_attributes(self):
        self.FT_magnitude()
        self.FT_phase()
        self.FT_imaginary()
        self.FT_real()
        
    def get_pix_maps(self):
        self.set_pix_magnitude()
        self.set_pix_phase()
        self.set_pix_imaginary()
        self.set_pix_real()
   
    def create_cutomized_image(self, size, minimum_width, minimum_height):
        if self.file_path != None:
            image = QtGui.QImage(self.file_path)
            # Convert the image to grayscale if necessary
            if image.format() != QtGui.QImage.Format_Grayscale8:
                image = image.convertToFormat(QtGui.QImage.Format_Grayscale8)
            # resize the pixmap to match the smallest image size
            if image.width() != minimum_width or image.height() != minimum_height:
                image = image.scaled(minimum_width, minimum_height)
            # Create a pixmap from the image
            pixmap = QtGui.QPixmap.fromImage(image)
            # Scale the pixmap to fit in the widget
            self.img = pixmap.scaled(size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        
class ImageProcessingThread(QThread):
    processing_done = pyqtSignal(int, np.ndarray)

    def get_image_data(self, image_data, index):
        self.image_data= image_data
        self.index= index
    def run(self):
        processed_data =np.real(np.fft.ifft2(np.fft.ifftshift(self.image_data)))
        # Emit signal to indicate processing is done
        self.processing_done.emit(self.index, processed_data)



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1350, 670)
        self.bright = 1
        self.contrast = 1.0
        self.images = [Image() for _ in range(4)]
        self.sizes=[math.inf for _ in range(4)]
        self.areas=[math.inf for _ in range(4)]
        self.minimum_area= math.inf
        self.processing_thread1 = ImageProcessingThread()
        self.processing_thread2 = ImageProcessingThread()
        self.processing_thread1.processing_done.connect(self.display_output)
        self.processing_thread2.processing_done.connect(self.display_output)
        self.start_point = None
        self.end_point = None
        self.current_rect = None
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMinimumSize(QtCore.QSize(100, 0))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.input_1 = QtWidgets.QWidget(self.frame)
        self.input_1.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.input_1.setObjectName("input_1")
        self.gridLayout.addWidget(self.input_1, 0, 0, 1, 1)
        self.input_1.mouseDoubleClickEvent = lambda event: self.browse_image(event, self.input_1)
        self.input_1.mousePressEvent = lambda event:self.on_mouse_press(event)
        self.input_1.mouseMoveEvent = lambda event: self.on_mouse_move(event, 1)
        self.fourier_1 = QtWidgets.QWidget(self.frame)
        self.fourier_1.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.fourier_1.setObjectName("fourier_1")
        self.gridLayout.addWidget(self.fourier_1, 1, 0, 1, 1)
        self.frame_6 = QtWidgets.QFrame(self.frame)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_6)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_23 = QtWidgets.QLabel(self.frame_6)
        self.label_23.setObjectName("label_23")
        self.gridLayout_6.addWidget(self.label_23, 1, 0, 1, 1)
        self.fourier_combobox1 = QtWidgets.QComboBox(self.frame_6)
        self.fourier_combobox1.setMinimumSize(QtCore.QSize(0, 20))
        self.fourier_combobox1.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fourier_combobox1.setObjectName("fourier_combobox1")
        self.fourier_combobox1.addItem("")
        self.fourier_combobox1.addItem("")
        self.fourier_combobox1.addItem("")
        self.fourier_combobox1.addItem("")
        self.gridLayout_6.addWidget(self.fourier_combobox1, 0, 0, 1, 2)
        self.fourier_combobox1.currentTextChanged.connect(lambda: self.display_fourier_transform(0, self.fourier_combobox1.currentText()))
        self.fourier_1.mousePressEvent = self.start_freq_value
        self.fourier_1.mouseMoveEvent = self.tracing_freq
        self.fourier_1.mouseReleaseEvent= self.end_freq_value
        
        self.magnitude_slider1 = QtWidgets.QSlider(self.frame_6)
        self.magnitude_slider1.setOrientation(QtCore.Qt.Horizontal)
        self.magnitude_slider1.setObjectName("magnitude_slider1")
        self.gridLayout_6.addWidget(self.magnitude_slider1, 1, 1, 1, 1)
        self.magnitude_slider1.setMaximum(100)
        self.magnitude_slider1.sliderReleased.connect(self.update_displayed_output)
        self.phase_slider1 = QtWidgets.QSlider(self.frame_6)
        self.phase_slider1.setOrientation(QtCore.Qt.Horizontal)
        self.phase_slider1.setObjectName("phase_slider1")
        self.phase_slider1.setMaximum(100)
        self.phase_slider1.sliderReleased.connect(self.update_displayed_output)
        self.gridLayout_6.addWidget(self.phase_slider1, 2, 1, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.frame_6)
        self.label_24.setObjectName("label_24")
        self.gridLayout_6.addWidget(self.label_24, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.frame_6, 2, 0, 1, 1)
        self.gridLayout_10.addWidget(self.frame, 0, 0, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setMinimumSize(QtCore.QSize(100, 0))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.input_2 = QtWidgets.QWidget(self.frame_2)
        self.input_2.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.input_2.setObjectName("input_2")
        self.gridLayout_2.addWidget(self.input_2, 0, 0, 1, 1)
        self.input_2.mouseDoubleClickEvent = lambda event: self.browse_image(event, self.input_2)
        self.input_2.mousePressEvent = lambda event:self.on_mouse_press(event)
        self.input_2.mouseMoveEvent = lambda event: self.on_mouse_move(event, 2)
        self.fourier_2 = QtWidgets.QWidget(self.frame_2)
        self.fourier_2.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.fourier_2.setObjectName("fourier_2")
        self.gridLayout_2.addWidget(self.fourier_2, 1, 0, 1, 1)
        self.fourier_2.mousePressEvent = self.start_freq_value
        self.fourier_2.mouseMoveEvent = self.tracing_freq
        self.fourier_2.mouseReleaseEvent= self.end_freq_value
        self.frame_7 = QtWidgets.QFrame(self.frame_2)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_7)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_26 = QtWidgets.QLabel(self.frame_7)
        self.label_26.setObjectName("label_26")
        self.gridLayout_7.addWidget(self.label_26, 2, 0, 1, 1)
        self.phase_slider2 = QtWidgets.QSlider(self.frame_7)
        self.phase_slider2.setOrientation(QtCore.Qt.Horizontal)
        self.phase_slider2.setObjectName("phase_slider2")
        self.phase_slider2.setMaximum(100)
        self.phase_slider2.sliderReleased.connect(self.update_displayed_output)
        self.gridLayout_7.addWidget(self.phase_slider2, 2, 1, 1, 1)
        self.magnitude_slider2 = QtWidgets.QSlider(self.frame_7)
        self.magnitude_slider2.setOrientation(QtCore.Qt.Horizontal)
        self.magnitude_slider2.setObjectName("magnitude_slider2")
        self.gridLayout_7.addWidget(self.magnitude_slider2, 1, 1, 1, 1)
        self.magnitude_slider2.setMaximum(100)
        self.magnitude_slider2.sliderReleased.connect(self.update_displayed_output)
        
        self.label_25 = QtWidgets.QLabel(self.frame_7)
        self.label_25.setObjectName("label_25")
        self.gridLayout_7.addWidget(self.label_25, 1, 0, 1, 1)
        self.fourier_combobox2 = QtWidgets.QComboBox(self.frame_7)
        self.fourier_combobox2.setMinimumSize(QtCore.QSize(0, 20))
        self.fourier_combobox2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fourier_combobox2.setObjectName("fourier_combobox2")
        self.fourier_combobox2.addItem("")
        self.fourier_combobox2.addItem("")
        self.fourier_combobox2.addItem("")
        self.fourier_combobox2.addItem("")
        self.fourier_combobox2.currentTextChanged.connect(lambda: self.display_fourier_transform(1, self.fourier_combobox2.currentText()))
        self.gridLayout_7.addWidget(self.fourier_combobox2, 0, 0, 1, 2)
        self.gridLayout_2.addWidget(self.frame_7, 2, 0, 1, 1)
        self.gridLayout_10.addWidget(self.frame_2, 0, 1, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setMinimumSize(QtCore.QSize(100, 0))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.fourier_3 = QtWidgets.QWidget(self.frame_3)
        self.fourier_3.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.fourier_3.setObjectName("fourier_3")
        self.gridLayout_3.addWidget(self.fourier_3, 1, 0, 1, 1)
        self.fourier_3.mousePressEvent = self.start_freq_value
        self.fourier_3.mouseMoveEvent = self.tracing_freq
        self.fourier_3.mouseReleaseEvent= self.end_freq_value
        self.input_3 = QtWidgets.QWidget(self.frame_3)
        self.input_3.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.input_3.setObjectName("input_3")
        self.gridLayout_3.addWidget(self.input_3, 0, 0, 1, 1)
        self.input_3.mouseDoubleClickEvent = lambda event: self.browse_image(event, self.input_3)
        self.input_3.mousePressEvent = lambda event:self.on_mouse_press(event)
        self.input_3.mouseMoveEvent = lambda event: self.on_mouse_move(event, 3)
        self.frame_8 = QtWidgets.QFrame(self.frame_3)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.frame_8)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.phase_slider3 = QtWidgets.QSlider(self.frame_8)
        self.phase_slider3.setOrientation(QtCore.Qt.Horizontal)
        self.phase_slider3.setObjectName("phase_slider3")
        self.phase_slider3.setMaximum(100)
        self.gridLayout_8.addWidget(self.phase_slider3, 2, 1, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.frame_8)
        self.label_27.setObjectName("label_27")
        self.phase_slider3.sliderReleased.connect(self.update_displayed_output)
        self.gridLayout_8.addWidget(self.label_27, 1, 0, 1, 1)
        self.fourier_combobox3 = QtWidgets.QComboBox(self.frame_8)
        self.fourier_combobox3.setMinimumSize(QtCore.QSize(0, 20))
        self.fourier_combobox3.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fourier_combobox3.setObjectName("fourier_combobox3")
        self.fourier_combobox3.addItem("")
        self.fourier_combobox3.addItem("")
        self.fourier_combobox3.addItem("")
        self.fourier_combobox3.addItem("")
        self.fourier_combobox3.currentTextChanged.connect(lambda: self.display_fourier_transform(2, self.fourier_combobox3.currentText()))
        self.gridLayout_8.addWidget(self.fourier_combobox3, 0, 0, 1, 2)
        
        self.label_28 = QtWidgets.QLabel(self.frame_8)
        self.label_28.setObjectName("label_28")
        self.gridLayout_8.addWidget(self.label_28, 2, 0, 1, 1)
        self.magnitude_slider3 = QtWidgets.QSlider(self.frame_8)
        self.magnitude_slider3.setOrientation(QtCore.Qt.Horizontal)
        self.magnitude_slider3.setObjectName("magnitude_slider3")
        self.magnitude_slider3.setMaximum(100)
        self.magnitude_slider3.sliderReleased.connect(self.update_displayed_output)
        self.gridLayout_8.addWidget(self.magnitude_slider3, 1, 1, 1, 1)
        self.gridLayout_3.addWidget(self.frame_8, 2, 0, 1, 1)
        self.gridLayout_10.addWidget(self.frame_3, 0, 2, 1, 1)
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_5)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.output_1 = QtWidgets.QWidget(self.frame_5)
        self.output_1.setMinimumSize(QtCore.QSize(300, 0))
        self.output_1.setMaximumSize(QtCore.QSize(300, 16777215))
        self.output_1.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.output_1.setObjectName("output_1")
        self.gridLayout_5.addWidget(self.output_1, 0, 0, 1, 2)
        self.output_2 = QtWidgets.QWidget(self.frame_5)
        self.output_2.setMinimumSize(QtCore.QSize(300, 0))
        self.output_2.setMaximumSize(QtCore.QSize(300, 16777215))
        self.output_2.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.output_2.setObjectName("output_2")
        self.gridLayout_5.addWidget(self.output_2, 1, 0, 1, 2)
        self.checkBox_output1 = QtWidgets.QCheckBox(self.frame_5)
        self.checkBox_output1.setObjectName("checkBox_output1")
        self.gridLayout_5.addWidget(self.checkBox_output1, 4, 0, 1, 1)
        self.checkBox_output1.setCheckState(True)
        self.checkBox_output2 = QtWidgets.QCheckBox(self.frame_5)
        self.checkBox_output2.setObjectName("checkBox_output2")
        self.checkBox_output2.clicked.connect(self.update_displayed_output)
        self.checkBox_output1.clicked.connect(self.update_displayed_output)
        self.gridLayout_5.addWidget(self.checkBox_output2, 4, 1, 1, 1)

        self.frequency_view = QtWidgets.QComboBox(self.frame_5)
        self.frequency_view.setObjectName("frequency_view")
        self.frequency_view.addItem("")
        self.frequency_view.addItem("")
        self.frequency_view.addItem("")
        self.frequency_view.setCurrentIndex(2)
        self.frequency_view.currentTextChanged.connect(self.recalculating)
        self.gridLayout_5.addWidget(self.frequency_view, 2, 0, 1, 2)
        self.display_combobox1 = QtWidgets.QComboBox(self.frame_6)
        self.display_combobox1.setMinimumSize(QtCore.QSize(0, 20))
        self.display_combobox1.setMaximumSize(QtCore.QSize(16777215, 20))
        self.display_combobox1.setObjectName("display_combobox1")
        self.display_combobox1.addItem("")
        self.display_combobox1.addItem("")
        self.gridLayout_5.addWidget(self.display_combobox1, 3, 0, 1, 2)
        self.display_combobox1.currentTextChanged.connect(self.change_slider_labels)
        self.gridLayout_10.addWidget(self.frame_5, 0, 4, 1, 1)
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setMinimumSize(QtCore.QSize(100, 0))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.input_4 = QtWidgets.QWidget(self.frame_4)
        self.input_4.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.input_4.setObjectName("input_4")
        self.input_4.mouseDoubleClickEvent = lambda event: self.browse_image(event, self.input_4)
        self.input_4.mousePressEvent = lambda event:self.on_mouse_press(event)
        self.input_4.mouseMoveEvent = lambda event: self.on_mouse_move(event, 4)
        self.gridLayout_4.addWidget(self.input_4, 0, 0, 1, 1)
        self.fourier_4 = QtWidgets.QWidget(self.frame_4)
        self.fourier_4.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.fourier_4.setObjectName("fourier_4")
        self.gridLayout_4.addWidget(self.fourier_4, 1, 0, 1, 1)
        self.fourier_4.mousePressEvent = self.start_freq_value
        self.fourier_4.mouseMoveEvent = self.tracing_freq
        self.fourier_4.mouseReleaseEvent= self.end_freq_value
        self.frame_9 = QtWidgets.QFrame(self.frame_4)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.frame_9)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.phase_slider4 = QtWidgets.QSlider(self.frame_9)
        self.phase_slider4.setOrientation(QtCore.Qt.Horizontal)
        self.phase_slider4.setObjectName("phase_slider4")
        self.gridLayout_9.addWidget(self.phase_slider4, 2, 1, 1, 1)
        self.phase_slider4.sliderReleased.connect(self.update_displayed_output)
        self.phase_slider4.setMaximum(100)
        self.label_30 = QtWidgets.QLabel(self.frame_9)
        self.label_30.setObjectName("label_30")
        self.gridLayout_9.addWidget(self.label_30, 2, 0, 1, 1)
        self.magnitude_slider4 = QtWidgets.QSlider(self.frame_9)
        self.magnitude_slider4.setOrientation(QtCore.Qt.Horizontal)
        self.magnitude_slider4.setObjectName("magnitude_slider4")
        self.magnitude_slider4.setMaximum(100)
        self.magnitude_slider4.sliderReleased.connect(self.update_displayed_output)
        self.gridLayout_9.addWidget(self.magnitude_slider4, 1, 1, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.frame_9)
        self.label_29.setObjectName("label_29")
        self.gridLayout_9.addWidget(self.label_29, 1, 0, 1, 1)
        self.fourier_combobox4 = QtWidgets.QComboBox(self.frame_9)
        self.fourier_combobox4.setMinimumSize(QtCore.QSize(0, 20))
        self.fourier_combobox4.setMaximumSize(QtCore.QSize(16777215, 20))
        self.fourier_combobox4.setObjectName("fourier_combobox4")
        self.fourier_combobox4.addItem("")
        self.fourier_combobox4.addItem("")
        self.fourier_combobox4.addItem("")
        self.fourier_combobox4.addItem("")
        self.fourier_combobox4.currentTextChanged.connect(lambda: self.display_fourier_transform(3, self.fourier_combobox4.currentText()))
        self.gridLayout_9.addWidget(self.fourier_combobox4, 0, 0, 1, 2)
        self.gridLayout_4.addWidget(self.frame_9, 2, 0, 1, 1)
        self.gridLayout_10.addWidget(self.frame_4, 0, 3, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def change_slider_labels(self):
        for i in range(23, 31, 2):
            label1= getattr(self, 'label_'+ str(i))
            label2= getattr(self, 'label_'+ str(i+1))
            if self.display_combobox1.currentText()=="Magnitude & Phase":
                label1.setText("Magnitude")
                label2.setText("Phase")
            else:
                label1.setText("Real")
                label2.setText("Imaginary")

    def browse_image(self, event, target_widget):
        # Open a file dialog to select an image file
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        file_dialog.setViewMode(QtWidgets.QFileDialog.Detail)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.file_path = selected_files[0]
                self.load_image(self.file_path, target_widget)
    
    def load_image(self, file_path, target_widget):
        # Remove existing QLabel from the layout if it exists
        if target_widget.layout() is not None:
            while target_widget.layout().count():
                item = target_widget.layout().takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        else:
            # Create a QVBoxLayout for the target_widget
            layout = QtWidgets.QVBoxLayout(target_widget)
            target_widget.setLayout(layout)

        for i in range(1, 5):
            if target_widget == getattr(self, f"input_{i}"):
                image = QtGui.QImage(file_path)
                index=i-1
                image_size= image.size()
                self.sizes[index]= image_size
                self.areas[index]= image_size.width() * image_size.height()
                if self.minimum_area> self.areas[index]:
                    self.set_minimum_size(self.areas[index], index, image_size.height(), image_size.width())
                    for i in range (4):
                        if self.images[i].img != None:
                            target= getattr(self, f"input_{i+1}")
                            self.images[i].get_image_attributes(target.size(), self.minimum_width, self.minimum_height)
                self.images[index].file_path= file_path
                self.images[index].get_image_attributes(target_widget.size(), self.minimum_width, self.minimum_height)
                self.update_displayed_output()
                self.display_img(target_widget, index)
                combobox= getattr(self, f"fourier_combobox{index+1}")
                if combobox.currentIndex() != -1:
                    text= combobox.currentText()
                    self.display_fourier_transform(index, text)
                break
        

    def set_minimum_size(self, minimum_area, index, minimum_height, minimum_width):
        self.minimum_area= minimum_area
        self.minimum_size= self.sizes[index]
        self.minimum_height= minimum_height
        self.minimum_width= minimum_width
    
    def display_img(self, target_widget, index):
        # Create a QLabel and set the scaled pixmap as its background
        label = QtWidgets.QLabel(target_widget)
        label.setPixmap(self.images[index].img)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setScaledContents(True)  

        # Add the label to the layout
        target_widget.layout().addWidget(label)
        target_widget.layout().setContentsMargins(0, 0, 0, 0)
        target_widget.layout().setSpacing(0)
    
    def display_fourier_transform(self, index, text): 
        if self.images[index].img != None:
            if text == "Magnitude":
                ft_output = self.images[index].pix_mag
            elif text == "Phase":
                ft_output = self.images[index].pix_phase
            elif text == "Real":
                ft_output = self.images[index].pix_real
            elif text == "Imaginary":
                ft_output = self.images[index].pix_imaginary
            fourier= getattr(self, "fourier_" + str(index+1))
            # Create a QLabel to display the QPixmap
            label = QLabel(fourier)
            scaled_pixmap = ft_output.scaled(self.minimum_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            if self.current_rect:
                painter = QPainter(scaled_pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                pen = QPen(Qt.green, 2, Qt.DashLine)
                brush = QBrush(Qt.NoBrush)
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawRect(self.current_rect)
                del painter 
            label.setPixmap(scaled_pixmap)
            label.setScaledContents(True)
            label.setGeometry(QtCore.QRect(0, 0, fourier.width(), fourier.height()))
            label.show()
        
    def update_displayed_output(self):
        for i in range(4):
            if not self.images[i].img == None:
                self.weighted_ft_mag=[np.zeros_like(self.images[i].mag) for _ in range(1)]  
                self.weighted_ft_phase=[np.zeros_like(self.images[i].phase) for _ in range(1)]
                self.weighted_ft= [np.zeros_like(self.images[i].phase) for _ in range(1)]
                break
        for i in range(4):
            if not self.images[i].img == None:
                # Get the weights from the sliders
                magnitude_weight = getattr(self, f"magnitude_slider{i+1}").value() / 100.0
                phase_weight = getattr(self, f"phase_slider{i+1}").value() / 100.0
                try:
                    if self.display_combobox1.currentIndex() ==0:
                        # Calculate the weighted average of the Fourier Transforms
                        mag_weighted_ft, phase_weighted_ft= self.add_weights(magnitude_weight,self.images[i].mag,phase_weight, self.images[i].phase )
                    else:
                        mag_weighted_ft, phase_weighted_ft =  self.add_weights(magnitude_weight,self.images[i].real, phase_weight, self.images[i].imaginary )

                    self.weighted_ft_mag[0]= np.add(mag_weighted_ft, self.weighted_ft_mag[0])
                    self.weighted_ft_phase[0]= np.add(phase_weighted_ft, self.weighted_ft_phase[0])
                except Exception as e:
                     print(f"Error processing image {i+1}: {str(e)}")
        if  self.display_combobox1.currentIndex() ==0:
            self.weighted_ft[0]= self.weighted_ft_mag[0] * np.exp(1j * self.weighted_ft_phase[0])
        else:
            self.weighted_ft[0]= self.weighted_ft_mag[0] + 1j * self.weighted_ft_phase[0]
        if self.checkBox_output1.isChecked():
            self.processing_thread1.get_image_data(self.weighted_ft[0], 0)
            self.processing_thread1.start()
        if self.checkBox_output2.isChecked():
            self.processing_thread2.get_image_data(self.weighted_ft[0], 1)
            self.processing_thread2.start()

    def add_weights(self, weight_array1, array1, weight_array2, array2):
        mag_weighted_ft =  array1 * weight_array1
        phase_weighted_ft= array2 * weight_array2
        return mag_weighted_ft, phase_weighted_ft
    def display_output(self, index, value):
        cv2.imwrite('test.jpg', value)
        image= cv2.imread(r'C:\Users\DELL\Downloads\test.jpg', cv2.IMREAD_GRAYSCALE)
        # Scale the output image for display
        output_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # Convert the output image array to QImage
        output_image_qimage = QImage(
            output_image.data,
            output_image.shape[1],
            output_image.shape[0],
            output_image.shape[1],
            QImage.Format_Grayscale8
        )
        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(output_image_qimage)
        out= getattr(self,f"output_{index+1}" )
        # Display the output image
        label = QLabel(out)
        label.setPixmap(pixmap.scaled(out.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        label.setScaledContents(True)
        label.setGeometry(QtCore.QRect(0, 0, out.width(), out.height()))
        label.show()

    def on_mouse_press(self, event):
        # Store the mouse click position
        self.mouse_click_pos = event.pos()


    def on_mouse_move(self, event, index):
        if self.mouse_click_pos is not None:
            # Calculate the mouse movement distance
            mouse_move_distance = event.pos() - self.mouse_click_pos

            # Check the direction of mouse movement
            if mouse_move_distance.y() > 0:
                # Mouse moved down (decrease brightness)
                self.bright -= 1
            elif mouse_move_distance.y() < 0:
                # Mouse moved up (increase brightness)
                self.bright += 1

            if mouse_move_distance.x() > 0:
                # Mouse moved right (increase contrast)
                self.contrast += 0.1
            elif mouse_move_distance.x() < 0:
                # Mouse moved left (decrease contrast)
                self.contrast -= 0.1

            # Apply the updated brightness and contrast to the image
            self.apply_brightness_contrast(index)

            # Store the current mouse position for the next movement calculation
            self.mouse_click_pos = event.pos()

    def apply_brightness_contrast(self, index):
        # Load the image
        image = cv2.imread(self.images[index-1].file_path)

        # Apply brightness and contrast adjustment
        adjusted_image = np.clip(self.contrast * image + self.bright, 0, 255).astype(np.uint8)

        # Convert the adjusted image to grayscale
        grayscale_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

        # Convert the numpy array to QImage
        height, width = grayscale_image.shape
        bytesPerLine = width
        q_image = QImage(grayscale_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)

        scaled_pixmap = pixmap.scaled(self.input_1.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        input_widget= getattr(self, f"input_{index}")
        # Create a QLabel and set the scaled pixmap as its background
        label = QtWidgets.QLabel(input_widget)
        label.setPixmap(scaled_pixmap)
        label.setScaledContents(True)
        label.setGeometry(QtCore.QRect(0, 0, self.input_1.width(), self.input_1.height()))
        label.show()
    
    def start_freq_value(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            

    def tracing_freq(self, event):
        if event.buttons() & Qt.LeftButton:
            self.end_point = event.pos()
            

    def end_freq_value(self, event):
        if event.button() == Qt.LeftButton:
            start_x, start_y = self.start_point.x(), self.start_point.y()
            end_x, end_y = self.end_point.x(), self.end_point.y()
            start_x*= (self.minimum_width/ self.fourier_1.width())
            end_x*= (self.minimum_width/ self.fourier_1.width())
            start_y*=(self.minimum_height/ self.fourier_1.height())
            end_y*= (self.minimum_height/self.fourier_1.height())
            self.range1= (int(start_y),int(end_y))
            self.range2=(int(start_x), int(end_x))
            self.current_rect = QRect(QPoint(int(start_x), int(start_y)), QPoint(int(end_x), int(end_y)))
            self.recalculating()
            self.start_point = None
            self.end_point = None
            for i in range(4):
                if self.images[i].img != None:
                    combobox= getattr(self, f"fourier_combobox{i+1}")
                    text= combobox.currentText()
                    self.display_fourier_transform(i, text)
            

    def recalculating (self):
        for i in range (4):
            if self.images[i].img != None:
                if self.current_rect:
                    self.images[i].fourier_transform(self.minimum_width, self.minimum_height)
                    temp_array = np.zeros_like(self.images[i].fourier)
                    if self.frequency_view.currentIndex() ==0:
                        temp_array[self.range1[0]:self.range1[1]+1, self.range2[0]:self.range2[1]+1] = self.images[i].fourier[self.range1[0]:self.range1[1]+1,self.range2[0]:self.range2[1]+1]
                        self.images[i].fourier= temp_array
                    elif self.frequency_view.currentIndex() ==1:
                        self.images[i].fourier[self.range1[0]:self.range1[1]+1,self.range2[0]:self.range2[1]+1] = temp_array[self.range1[0]:self.range1[1]+1, self.range2[0]:self.range2[1]+1]
                        
                    self.images[i].get_output_related_attributes()
        if self.current_rect:
            self.update_displayed_output()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_23.setText(_translate("MainWindow", "Magnitude"))
        self.fourier_combobox1.setItemText(0, _translate("MainWindow", "Magnitude"))
        self.fourier_combobox1.setItemText(1, _translate("MainWindow", "Phase"))
        self.fourier_combobox1.setItemText(2, _translate("MainWindow", "Real"))
        self.fourier_combobox1.setItemText(3, _translate("MainWindow", "Imaginary"))
        self.display_combobox1.setItemText(0, _translate("MainWindow", "Magnitude & Phase"))
        self.display_combobox1.setItemText(1, _translate("MainWindow", "Real & Imaginary"))
        self.label_24.setText(_translate("MainWindow", "Phase"))
        self.label_26.setText(_translate("MainWindow", "Phase"))
        self.label_25.setText(_translate("MainWindow", "Magnitude"))
        self.fourier_combobox2.setItemText(0, _translate("MainWindow", "Magnitude"))
        self.fourier_combobox2.setItemText(1, _translate("MainWindow", "Phase"))
        self.fourier_combobox2.setItemText(2, _translate("MainWindow", "Real"))
        self.fourier_combobox2.setItemText(3, _translate("MainWindow", "Imaginary"))
        self.label_27.setText(_translate("MainWindow", "Magnitude"))
        self.fourier_combobox3.setItemText(0, _translate("MainWindow", "Magnitude"))
        self.fourier_combobox3.setItemText(1, _translate("MainWindow", "Phase"))
        self.fourier_combobox3.setItemText(2, _translate("MainWindow", "Real"))
        self.fourier_combobox3.setItemText(3, _translate("MainWindow", "Imaginary"))
        self.label_28.setText(_translate("MainWindow", "Phase"))
        self.checkBox_output1.setText(_translate("MainWindow", "Output 1"))
        self.checkBox_output2.setText(_translate("MainWindow", "Output 2"))
        self.frequency_view.setItemText(0, _translate("MainWindow", "InnerFrequency"))
        self.frequency_view.setItemText(1, _translate("MainWindow", "Outer Frequency"))
        self.frequency_view.setItemText(2, _translate("MainWindow", "Full Frequency"))
        self.label_30.setText(_translate("MainWindow", "Phase"))
        self.label_29.setText(_translate("MainWindow", "Magnitude"))
        self.fourier_combobox4.setItemText(0, _translate("MainWindow", "Magnitude"))
        self.fourier_combobox4.setItemText(1, _translate("MainWindow", "Phase"))
        self.fourier_combobox4.setItemText(2, _translate("MainWindow", "Real"))
        self.fourier_combobox4.setItemText(3, _translate("MainWindow", "Imaginary"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
