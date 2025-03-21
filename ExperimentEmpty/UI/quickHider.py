# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QuickSearcher.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5 import QtCore, QtGui, QtWidgets
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from sensor_msgs.msg import Image
from std_msgs.msg import Int16, String
import math
import numpy as np
import time
import random 
import itertools

import cv2

import cv2
from cv_bridge import CvBridge


import numpy as np
import cv2

from PyQt5 import *
from PyQt5.QtCore import QTimer
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

strats = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]


class Ui_View(QWidget):
    def __init__(self):
        super().__init__()
        self.disply_width = 791
        self.display_height = 480

        

        
        self.br = CvBridge()

        # create the video capture thread
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(5)

        self.allH = ['naive', 'robot']

        self.runs = 1
        self.succ = 0
        self.hidP = [0,0,0]
        self.stra = [0,0,0,0,0,0]



        rclpy.init(args=None)

        self.sub_Node = Node('Image')
        self.sta = self.sub_Node.create_subscription(
            String, '/status',self.status_callback, 10)
        self.opponent = self.sub_Node.create_subscription(
            String, 'opponentS',self.opponent_callback, 10)
        self.button_publisher = self.sub_Node.create_publisher(Int16, '/hidder', 10)
        self.prev = self.sub_Node.create_subscription(
            String, 'prev',self.previous_callback, 10)

        self.sub2 = self.sub_Node.create_subscription(
            String, '/score',self.score_callback, 10)

    def update(self):
        rclpy.spin_once(self.sub_Node)

    def status_callback(self, msgs):
        text = msgs.data
        self.statusLabel.setText(text)
        self.successes.setText(str(self.succ))
        self.totalRuns.setText(str(self.runs))

    def score_callback(self, msg):
        score = msg.data
        score = score.split(',')
        self.stra = score[0:6]
        self.hidP = score[6:9]
        self.runs = score[9]
        self.succ = score[10]

        _translate = QtCore.QCoreApplication.translate
        for i in range(len(self.hidP)):
            item = self.hid_label.item(i)
            item.setText(_translate("View", f'hid at Spot {i+1}: x{self.hidP[i]}'))
        
        for i in range(len(self.stra)):
            item = self.strats_label.item(i)
            item.setText(_translate("View", f'{strats[i][0]+1}->{strats[i][1]+1}->{strats[i][2]+1}  used x{self.stra[i]}'))

    def opponent_callback(self, msg):
        op = msg.data
        if op == 'naive1':
            op = 'naive risky'
        elif op == 'naive2':
            op = 'naive safe'
        elif op == 'robot':
            op = 'belief-based'
        
        self.opponent_label.setText(f"You are playing against a {op} searcher")

    def previous_callback(self,msg):
        prev = msg.data
        self.prevS_label.setText(prev)




    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def clicked1(self):
        msg = Int16()
        msg.data = 0
        self.button_publisher.publish(msg)
    
    def clicked2(self):
        msg = Int16()
        msg.data = 1
        self.button_publisher.publish(msg)

    def clicked3(self):
        msg = Int16()
        msg.data = 2
        self.button_publisher.publish(msg)
    


        
    def setupUi(self, View):
        View.setObjectName("View")
        View.resize(790, 672)
        font = self.font()
        font.setPointSize(24)
        font2 = self.font()
        font2.setPointSize(16)
        self.centralwidget = QtWidgets.QWidget(View)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 230, 221, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(280, 230, 221, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(560, 230, 221, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(220, 160, 66, 19))
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(-10, 140, 791, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.statusLabel = QtWidgets.QLabel(self.centralwidget)
        self.statusLabel.setGeometry(QtCore.QRect(300, 160, 371, 19))
        self.statusLabel.setObjectName("statusLabel")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 450, 121, 19))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 430, 121, 19))
        self.label_4.setObjectName("label_4")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(-10, 410, 801, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.hidd_label = QtWidgets.QLabel(self.centralwidget)
        self.hidd_label.setGeometry(QtCore.QRect(590, 430, 141, 20))
        self.hidd_label.setObjectName("hidd_label")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(560, 420, 20, 191))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.totalRuns = QtWidgets.QLabel(self.centralwidget)
        self.totalRuns.setGeometry(QtCore.QRect(160, 430, 66, 19))
        self.totalRuns.setObjectName("totalRuns")
        self.successes = QtWidgets.QLabel(self.centralwidget)
        self.successes.setGeometry(QtCore.QRect(160, 450, 66, 19))
        self.successes.setObjectName("successes")
        self.hid_label = QtWidgets.QListWidget(self.centralwidget)
        self.hid_label.setGeometry(QtCore.QRect(580, 460, 201, 121))
        self.hid_label.setObjectName("hid_label")
        item = QtWidgets.QListWidgetItem()
        self.hid_label.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.hid_label.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.hid_label.addItem(item)
        self.Strategies = QtWidgets.QLabel(self.centralwidget)
        self.Strategies.setGeometry(QtCore.QRect(350, 210, 111, 19))
        self.Strategies.setObjectName("Strategies")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(0, 190, 791, 20))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.opponent_label = QtWidgets.QLabel(self.centralwidget)
        self.opponent_label.setGeometry(QtCore.QRect(0, 0, 781, 71))
        self.opponent_label.setLineWidth(20)
        self.opponent_label.setScaledContents(False)
        self.opponent_label.setAlignment(QtCore.Qt.AlignCenter)
        self.opponent_label.setObjectName("opponent_label")
        self.opponent_label.setFont(font)
        self.prevS_label = QtWidgets.QLabel(self.centralwidget)
        self.prevS_label.setGeometry(QtCore.QRect(0, 70, 781, 71))
        self.prevS_label.setLineWidth(20)
        self.prevS_label.setScaledContents(False)
        self.prevS_label.setAlignment(QtCore.Qt.AlignCenter)
        self.prevS_label.setObjectName("prevS_label")
        self.prevS_label.setFont(font2)
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(340, 420, 20, 191))
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.strats_label = QtWidgets.QListWidget(self.centralwidget)
        self.strats_label.setGeometry(QtCore.QRect(360, 460, 201, 121))
        self.strats_label.setObjectName("strats_label")
        item = QtWidgets.QListWidgetItem()
        self.strats_label.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.strats_label.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.strats_label.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.strats_label.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.strats_label.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.strats_label.addItem(item)
        self.stratsU_lable = QtWidgets.QLabel(self.centralwidget)
        self.stratsU_lable.setGeometry(QtCore.QRect(360, 430, 141, 20))
        self.stratsU_lable.setObjectName("stratsU_lable")
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setGeometry(QtCore.QRect(-10, 340, 801, 20))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 380, 171, 31))
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(300, 350, 181, 31))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(330, 380, 171, 31))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(610, 380, 171, 31))
        self.label_7.setObjectName("label_7")
        View.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(View)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 790, 24))
        self.menubar.setObjectName("menubar")
        View.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(View)
        self.statusbar.setObjectName("statusbar")
        View.setStatusBar(self.statusbar)

        self.pushButton.clicked.connect(self.clicked1)
        self.pushButton_2.clicked.connect(self.clicked2)
        self.pushButton_3.clicked.connect(self.clicked3)

        self.retranslateUi(View)
        QtCore.QMetaObject.connectSlotsByName(View)

    def retranslateUi(self, View):
        _translate = QtCore.QCoreApplication.translate
        View.setWindowTitle(_translate("View", "Human Search Trial"))
        self.pushButton.setText(_translate("View", "Spot 1"))
        self.pushButton_2.setText(_translate("View", "Spot 2"))
        self.pushButton_3.setText(_translate("View", "Spot 3"))
        self.label.setText(_translate("View", "Status:"))
        self.statusLabel.setText(_translate("View", "Currently Searching Spot 1"))
        self.label_3.setText(_translate("View", "Hiders found:"))
        self.label_4.setText(_translate("View", "Amount of Runs"))
        self.hidd_label.setText(_translate("View", "Hiding Spots:"))
        self.totalRuns.setText(_translate("View", "TextLabel"))
        self.successes.setText(_translate("View", "TextLabel"))
        
        self.Strategies.setText(_translate("View", "Strategies:"))
        self.opponent_label.setText(_translate("View", "Status:"))
        self.prevS_label.setText(_translate("View", "Status:"))
        __sortingEnabled = self.strats_label.isSortingEnabled()
        self.strats_label.setSortingEnabled(False)

        for i in range(len(self.hidP)):
            item = self.hid_label.item(i)
            item.setText(_translate("View", f'hid at Spot {i+1}: x{self.hidP[i]}'))
        self.strats_label.setSortingEnabled(__sortingEnabled)

        __sortingEnabled = self.hid_label.isSortingEnabled()
        self.hid_label.setSortingEnabled(False)
        
        for i in range(len(self.stra)):
            item = self.strats_label.item(i)
            item.setText(_translate("View", f'{strats[i][0]+1}->{strats[i][1]+1}->{strats[i][2]+1}  used x{self.stra[i]}'))
        self.hid_label.setSortingEnabled(__sortingEnabled)

        self.stratsU_lable.setText(_translate("View", "Strategies used:"))
        self.label_2.setText(_translate("View", "Spot 1:  50%"))
        self.label_5.setText(_translate("View", "Chances to being detected"))
        self.label_6.setText(_translate("View", "Spot 2:  75%"))
        self.label_7.setText(_translate("View", "Spot 3:  66%"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    View = QtWidgets.QMainWindow()
    ui = Ui_View()
    ui.setupUi(View)
    View.show()
    sys.exit(app.exec_())
