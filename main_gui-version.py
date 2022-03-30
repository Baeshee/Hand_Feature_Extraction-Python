import sys
import numpy as np
import cv2 as cv
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot, QModelIndex
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMainWindow,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QCheckBox,
    QPushButton
)

from PyQt6.QtGui import QPixmap, QColor, QImage

import os
import pandas as pd
from datetime import datetime as dt
from hand_detect_module import HandDetect

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def run(self):
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ret, img = cap.read()
            global cv_img
            cv_img = img
            if ret:
                self.change_pixmap_signal.emit(img)

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = HandDetect(detectCon=0.8, maxHands=2)
        

        self.setWindowTitle("Hand Feature Extraction")
        self.display_width = 1280
        self.display_height = 720
        width = 1280
        height = 720
        
        # Defining layout widgets
        h_lay = QHBoxLayout()
        v_lay_left = QVBoxLayout()
        v_lay_right = QVBoxLayout()
        
        tw_layout = QVBoxLayout()
        
        bw_layout = QHBoxLayout()
        bw_layout_left = QVBoxLayout()
        bw_layout_right = QVBoxLayout()
        
        # Top Widget
        topWidget = QWidget()
        topWidget.setFixedSize(QSize(200, 120))
        
        l_pn = QLabel('Testpersoon (nummer)')
        self.pn = QSpinBox()
        l_ht = QLabel('Hand die gefotografeerd wordt')
        self.ht = QComboBox()
        self.ht.addItems({"", "Left", "Right"})
        
        # Bottom Widget
        bottomWidget = QWidget()
        bottomWidget.setFixedSize(QSize(200, 300))
        
        self.r_btn = QPushButton('Rechter zijde')
        self.r_btn.clicked.connect(self.createAnnotedImage)
        self.t_btn = QPushButton('Bovenkant')
        self.t_btn.clicked.connect(self.createAnnotedImage)
        self.f_btn = QPushButton('Voorkant')
        self.f_btn.clicked.connect(self.createAnnotedImage)
        self.l_btn = QPushButton('Linker zijde')
        self.l_btn.clicked.connect(self.createAnnotedImage)
        
        self.r_cb = QCheckBox()
        self.t_cb = QCheckBox()
        self.f_cb = QCheckBox()
        self.l_cb = QCheckBox()
        
        # Adding widgets to layout
        tw_layout.addWidget(l_pn)
        tw_layout.addWidget(self.pn)
        tw_layout.addWidget(l_ht)
        tw_layout.addWidget(self.ht)
        
        bw_layout_left.addWidget(self.r_btn)
        bw_layout_left.addWidget(self.t_btn)
        bw_layout_left.addWidget(self.f_btn)
        bw_layout_left.addWidget(self.l_btn)
        bw_layout_left.insertStretch( -1, 1 );
        bw_layout_left.setSpacing(15)
        
        bw_layout_right.addWidget(self.r_cb)
        bw_layout_right.addWidget(self.t_cb)
        bw_layout_right.addWidget(self.f_cb)
        bw_layout_right.addWidget(self.l_cb)
        bw_layout_right.insertStretch( -1, 1 );
        bw_layout_right.setSpacing(25)
        bw_layout_right.setContentsMargins(10, 5, 0, 0)
        
        bw_layout.addLayout(bw_layout_left)
        bw_layout.addLayout(bw_layout_right)
        bw_layout.setContentsMargins(40, 0, 0, 0)
        
        topWidget.setLayout(tw_layout)
        bottomWidget.setLayout(bw_layout)

        v_lay_left.addWidget(topWidget)
        v_lay_left.addWidget(bottomWidget)
        
        self.image_label = QLabel(self)
        grey = QPixmap(width, height)
        grey.fill(QColor('darkGray'))
        self.image_label.setPixmap(grey)
        
        v_lay_right.addWidget(self.image_label)
        
        # Adding layout widgets
        h_lay.setSpacing(40)
        h_lay.addLayout(v_lay_left)
        h_lay.addLayout(v_lay_right)
        
        widget = QWidget()
        widget.setLayout(h_lay)

        # # Set the central widget of the Window. Widget will expand
        # # to take up all the space in the window by default.
        self.setCentralWidget(widget)
        
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        
    def createAnnotedImage(self):    
        sender = self.sender()
        side = ""
        
        if(sender.text() == "Rechter zijde"):
            self.r_cb.setChecked(True)
            side = "right"
        if(sender.text() == "Bovenkant"):
            self.t_cb.setChecked(True)
            side = "top"
        if(sender.text() == "Voorkant"):
            self.f_cb.setChecked(True)
            side = "front"
        if(sender.text() == "Linker zijde"):
            self.l_cb.setChecked(True)
            side = "left"
        
        start_time = dt.now()
        start_date = start_time.strftime("%d-%m-%Y")
        hands, img = self.detector.staticImage(cv_img)
        end_time = dt.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create a image where the landmarks are stored
        cv.imwrite(f"images/PY_testpersoon_{self.pn.value()}_{self.ht.currentText().lower()}_{side}_{start_date}.png", img)
        
        dict = {}
        hTypes = []
        hScore = []
        lmNames = []
        lmIds = []
        xList = []
        yList = []
        zList = []
        
        # Extracting all information that are stored in the variables
        # so they can be saved to a csv and excel for later use
        if hands:
            for h in range(0, len(hands)):
                for v in range(0, len(hands[0]['lmList'])):
                    hTypes.append(hands[h]['type'])
                    hScore.append(hands[h]['score'])
                    lmNames.append(hands[h]['lmList'][v]['name'])
                    lmIds.append(hands[h]['lmList'][v]['id'])
                    xList.append(hands[h]['lmList'][v]['x_value'])
                    yList.append(hands[h]['lmList'][v]['y_value'])
                    zList.append(hands[h]['lmList'][v]['z_value'])        
        
        dict = {'hand_type': hTypes,
                'hand_score': hScore,
                'landmark_id': lmIds,
                'landmark_name': lmNames,
                'x_value': xList,
                'y_value': yList,
                'z_value': zList,
                'total_run_time': duration,
                'test_date': dt.now().strftime("%d-%m-%Y"),
                'test_time': dt.now().strftime("%H:%M:%S"),
                'annoted_image': f"PY_testpersoon_{self.pn.value()}_{self.ht.currentText().lower()}_{side}_{start_date}.png"}
        
        # Check if there is a csv and a xslx file otherwise
        # create a new dataframe and save all the values to the
        # dataframe so they can be saved
        
        df = pd.DataFrame(dict)
        df.to_csv(f'results/PY_testpersoon_{self.pn.value()}_{self.ht.currentText().lower()}_{side}_{start_date}.csv', encoding='utf-8', index=False)
        
    @pyqtSlot(np.ndarray)
    def update_image(self, img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)


app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec()