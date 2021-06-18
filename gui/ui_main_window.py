from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from gui._init_variables import init_variables
from gui._birds_eye_config import (
    load_birds_eye_config_source,
    draw_point,
    reload_birds_view_image,
)
from gui._ref_obj_config import (
    load_ref_obj_config_src,
    reload_obj_ref_img,
    draw_ref_point,
)
from gui._config_buttons import (
    source_btn_clicked,
    show_mask_clicked,
    show_score_clicked,
    score_thresh_clicked,
    cam_source_clicked,
    iou_thresh_clicked,
    social_distance_clicked,
    configure_birds_eye,
    configure_ref_obj,
)
from gui._page_buttons import (
    yolo_btn_clicked,
    yolo_tiny_btn_clicked,
    load_weights_page_btn_clicked,
    birdseye_page_btn_clicked,
    ref_obj_page_btn,
)
from gui._general import (
    load_detection_source,
    pause_detection,
    update_ref_obj_dim,
)
from gui._detect_mask import detect_mask
from gui._detect_distance import detect_distance
from gui._detect_all import detect_all
from gui._load_weights import load_mask_weights, load_distance_weights


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        init_variables(self)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 850)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 850))
        MainWindow.setMouseTracking(False)
        MainWindow.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: #282a36;")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.centralwidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.centralwidget.setAcceptDrops(False)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_20.setSpacing(0)
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName("stackedWidget")
        self.detectionAlgorithmPage = QtWidgets.QWidget()
        self.detectionAlgorithmPage.setObjectName("detectionAlgorithmPage")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(
            self.detectionAlgorithmPage
        )
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.frame_7 = QtWidgets.QFrame(self.detectionAlgorithmPage)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.frame_13 = QtWidgets.QFrame(self.frame_7)
        self.frame_13.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_13.setStyleSheet("background-color: #44475a;")
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame_13)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_4 = QtWidgets.QLabel(self.frame_13)
        self.label_4.setStyleSheet(
            "color: #ff79c6;\n" 'font: 16pt "MS Shell Dlg 2";'
        )
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_8.addWidget(
            self.label_4, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_9.addWidget(self.frame_13)
        self.frame_14 = QtWidgets.QFrame(self.frame_7)
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.frame_14)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.yoloButton = QtWidgets.QPushButton(self.frame_14)
        self.yoloButton.setMaximumSize(QtCore.QSize(500, 16777215))
        self.yoloButton.setStyleSheet(
            "QPushButton#yoloButton {\n"
            "    color: #f8f8f2;\n"
            '    font: 16pt "MS Shell Dlg 2";\n'
            "    padding: 50px;\n"
            "    border: 1px solid #50fa7b;\n"
            "    border-radius: 5px\n"
            "}\n"
            "\n"
            "QPushButton#yoloButton:hover {\n"
            "    color: #50fa7b;\n"
            "}"
        )
        self.yoloButton.setObjectName("yoloButton")
        self.verticalLayout_10.addWidget(
            self.yoloButton, 0, QtCore.Qt.AlignHCenter
        )
        self.yoloTinyButton = QtWidgets.QPushButton(self.frame_14)
        self.yoloTinyButton.setMaximumSize(QtCore.QSize(500, 16777215))
        self.yoloTinyButton.setStyleSheet(
            "QPushButton#yoloTinyButton {\n"
            "    color: #f8f8f2;\n"
            '    font: 16pt "MS Shell Dlg 2";\n'
            "    padding: 50px;\n"
            "    border: 1px solid #50fa7b;\n"
            "    border-radius: 5px\n"
            "}\n"
            "\n"
            "QPushButton#yoloTinyButton:hover {\n"
            "    color: #50fa7b;\n"
            "}"
        )
        self.yoloTinyButton.setObjectName("yoloTinyButton")
        self.verticalLayout_10.addWidget(
            self.yoloTinyButton, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_9.addWidget(self.frame_14)
        self.verticalLayout_8.addWidget(self.frame_7)
        self.stackedWidget.addWidget(self.detectionAlgorithmPage)
        self.loadWeightsPage = QtWidgets.QWidget()
        self.loadWeightsPage.setObjectName("loadWeightsPage")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.loadWeightsPage)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame = QtWidgets.QFrame(self.loadWeightsPage)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_2.setStyleSheet("background-color: #44475a;")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setStyleSheet(
            "color: #ff79c6;\n" 'font: 16pt "MS Shell Dlg 2";'
        )
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_4.addWidget(self.frame_2)
        self.maskWeightesFrame = QtWidgets.QFrame(self.frame)
        self.maskWeightesFrame.setMaximumSize(QtCore.QSize(16777215, 100))
        self.maskWeightesFrame.setStyleSheet(
            "QFrame#maskWeightesFrane {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}\n"
            ""
        )
        self.maskWeightesFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.maskWeightesFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.maskWeightesFrame.setObjectName("maskWeightesFrame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.maskWeightesFrame)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.maskWeightsBtn = QtWidgets.QPushButton(self.maskWeightesFrame)
        self.maskWeightsBtn.setMinimumSize(QtCore.QSize(0, 50))
        self.maskWeightsBtn.setStyleSheet(
            "QPushButton#maskWeightsBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    border: 1px solid #6272a4;\n"
            "    border-radius: 5px;\n"
            "    margin: 5px 15px;\n"
            "}\n"
            "\n"
            "QPushButton#maskWeightsBtn:hover {\n"
            "    background-color: #6272a4;\n"
            "}\n"
            "\n"
            "QPushButton#maskWeightsBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}\n"
            "\n"
            ""
        )
        self.maskWeightsBtn.setObjectName("maskWeightsBtn")
        self.horizontalLayout_3.addWidget(self.maskWeightsBtn)
        self.maskPathLabel = QtWidgets.QLabel(self.maskWeightesFrame)
        self.maskPathLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #bd93f9;"
        )
        self.maskPathLabel.setObjectName("maskPathLabel")
        self.horizontalLayout_3.addWidget(
            self.maskPathLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.maskLoadedLabel = QtWidgets.QLabel(self.maskWeightesFrame)
        self.maskLoadedLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #bd93f9;"
        )
        self.maskLoadedLabel.setObjectName("maskLoadedLabel")
        self.horizontalLayout_3.addWidget(
            self.maskLoadedLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_4.addWidget(self.maskWeightesFrame)
        self.distanceWeightsFrame = QtWidgets.QFrame(self.frame)
        self.distanceWeightsFrame.setMaximumSize(QtCore.QSize(16777215, 100))
        self.distanceWeightsFrame.setStyleSheet(
            "QFrame#distanceWeightsFrame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}\n"
            ""
        )
        self.distanceWeightsFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.distanceWeightsFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.distanceWeightsFrame.setObjectName("distanceWeightsFrame")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(
            self.distanceWeightsFrame
        )
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.distanceWeightsBtn = QtWidgets.QPushButton(
            self.distanceWeightsFrame
        )
        self.distanceWeightsBtn.setMinimumSize(QtCore.QSize(0, 50))
        self.distanceWeightsBtn.setStyleSheet(
            "QPushButton#distanceWeightsBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    border: 1px solid #6272a4;\n"
            "    border-radius: 5px;\n"
            "    margin: 5px 15px;\n"
            "    padding-left: 5px;\n"
            "    padding-right: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#distanceWeightsBtn:hover {\n"
            "    background-color: #6272a4;\n"
            "}\n"
            "\n"
            "QPushButton#distanceWeightsBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.distanceWeightsBtn.setObjectName("distanceWeightsBtn")
        self.horizontalLayout_4.addWidget(self.distanceWeightsBtn)
        self.distancePathLabel = QtWidgets.QLabel(self.distanceWeightsFrame)
        self.distancePathLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #bd93f9;"
        )
        self.distancePathLabel.setObjectName("distancePathLabel")
        self.horizontalLayout_4.addWidget(
            self.distancePathLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.distanceLoadedLabel = QtWidgets.QLabel(self.distanceWeightsFrame)
        self.distanceLoadedLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #bd93f9;"
        )
        self.distanceLoadedLabel.setObjectName("distanceLoadedLabel")
        self.horizontalLayout_4.addWidget(
            self.distanceLoadedLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_4.addWidget(self.distanceWeightsFrame)
        self.frame_11 = QtWidgets.QFrame(self.frame)
        self.frame_11.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_11)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.loadWeightsPageBtn = QtWidgets.QPushButton(self.frame_11)
        self.loadWeightsPageBtn.setMaximumSize(QtCore.QSize(500, 16777215))
        self.loadWeightsPageBtn.setStyleSheet(
            "QPushButton#loadWeightsPageBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 16pt "MS Shell Dlg 2";\n'
            "    padding: 50px;\n"
            "    border: 1px solid #50fa7b;\n"
            "    border-radius: 5px\n"
            "}\n"
            "\n"
            "QPushButton#loadWeightsPageBtn:hover {\n"
            "    color: #50fa7b;\n"
            "}"
        )
        self.loadWeightsPageBtn.setObjectName("loadWeightsPageBtn")
        self.horizontalLayout_2.addWidget(self.loadWeightsPageBtn)
        self.verticalLayout_4.addWidget(self.frame_11)
        self.verticalLayout_3.addWidget(self.frame)
        self.stackedWidget.addWidget(self.loadWeightsPage)
        self.refObjConfigPage = QtWidgets.QWidget()
        self.refObjConfigPage.setObjectName("refObjConfigPage")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout(self.refObjConfigPage)
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_17.setSpacing(0)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.frame_15 = QtWidgets.QFrame(self.refObjConfigPage)
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.frame_15)
        self.verticalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.frame_16 = QtWidgets.QFrame(self.frame_15)
        self.frame_16.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_16.setStyleSheet("background-color: #44475a;")
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.frame_16)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_3 = QtWidgets.QLabel(self.frame_16)
        self.label_3.setStyleSheet(
            "color: #ff79c6;\n" 'font: 16pt "MS Shell Dlg 2";'
        )
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_11.addWidget(
            self.label_3, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_15.addWidget(self.frame_16)
        self.distanceWeightsFrame_3 = QtWidgets.QFrame(self.frame_15)
        self.distanceWeightsFrame_3.setMaximumSize(QtCore.QSize(16777215, 100))
        self.distanceWeightsFrame_3.setStyleSheet(
            "QFrame#distanceWeightsFrame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}\n"
            ""
        )
        self.distanceWeightsFrame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.distanceWeightsFrame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.distanceWeightsFrame_3.setObjectName("distanceWeightsFrame_3")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(
            self.distanceWeightsFrame_3
        )
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.referenceObjectLoadButton = QtWidgets.QPushButton(
            self.distanceWeightsFrame_3
        )
        self.referenceObjectLoadButton.setMinimumSize(QtCore.QSize(0, 50))
        self.referenceObjectLoadButton.setStyleSheet(
            "QPushButton#referenceObjectLoadButton {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    border: 1px solid #6272a4;\n"
            "    border-radius: 5px;\n"
            "    margin: 5px 15px;\n"
            "    padding-left: 5px;\n"
            "    padding-right: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#referenceObjectLoadButton:hover {\n"
            "    background-color: #6272a4;\n"
            "}\n"
            "\n"
            "QPushButton#referenceObjectLoadButton:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.referenceObjectLoadButton.setObjectName(
            "referenceObjectLoadButton"
        )
        self.horizontalLayout_13.addWidget(self.referenceObjectLoadButton)
        self.refObjLoadText = QtWidgets.QTextEdit(self.distanceWeightsFrame_3)
        self.refObjLoadText.setMaximumSize(QtCore.QSize(350, 50))
        self.refObjLoadText.setStyleSheet(
            "margin-left: 5px;\n"
            "color: #ffb86c;\n"
            'font: 12pt "MS Shell Dlg 2";\n'
            "border: 0.5px solid #6272a4;\n"
            "border-radius: 5px;\n"
            "padding-top: 10px;"
        )
        self.refObjLoadText.setObjectName("refObjLoadText")
        self.horizontalLayout_13.addWidget(
            self.refObjLoadText, 0, QtCore.Qt.AlignHCenter
        )
        self.refObjLoadLabel = QtWidgets.QLabel(self.distanceWeightsFrame_3)
        self.refObjLoadLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #bd93f9;"
        )
        self.refObjLoadLabel.setObjectName("refObjLoadLabel")
        self.horizontalLayout_13.addWidget(
            self.refObjLoadLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_15.addWidget(self.distanceWeightsFrame_3)
        self.refObjOutputFrame = QtWidgets.QFrame(self.frame_15)
        self.refObjOutputFrame.setMinimumSize(QtCore.QSize(416, 416))
        self.refObjOutputFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.refObjOutputFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.refObjOutputFrame.setObjectName("refObjOutputFrame")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.refObjOutputFrame)
        self.verticalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_16.setSpacing(0)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.refObjOutputLabel = QtWidgets.QLabel(self.refObjOutputFrame)
        self.refObjOutputLabel.setText("")
        self.refObjOutputLabel.setObjectName("refObjOutputLabel")
        self.verticalLayout_16.addWidget(
            self.refObjOutputLabel,
            0,
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
        )
        self.verticalLayout_15.addWidget(self.refObjOutputFrame)
        self.frame_17 = QtWidgets.QFrame(self.frame_15)
        self.frame_17.setMaximumSize(QtCore.QSize(16777215, 150))
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout(self.frame_17)
        self.horizontalLayout_22.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_22.setSpacing(0)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.refObjConfigPageBtn = QtWidgets.QPushButton(self.frame_17)
        self.refObjConfigPageBtn.setMaximumSize(QtCore.QSize(500, 16777215))
        self.refObjConfigPageBtn.setStyleSheet(
            "QPushButton#refObjConfigPageBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 16pt "MS Shell Dlg 2";\n'
            "    padding: 50px;\n"
            "    border: 1px solid #50fa7b;\n"
            "    border-radius: 5px\n"
            "}\n"
            "\n"
            "QPushButton#refObjConfigPageBtn:hover {\n"
            "    color: #50fa7b;\n"
            "}"
        )
        self.refObjConfigPageBtn.setObjectName("refObjConfigPageBtn")
        self.horizontalLayout_22.addWidget(self.refObjConfigPageBtn)
        self.verticalLayout_15.addWidget(self.frame_17)
        self.verticalLayout_17.addWidget(self.frame_15)
        self.stackedWidget.addWidget(self.refObjConfigPage)
        self.birdseyeConfigPage = QtWidgets.QWidget()
        self.birdseyeConfigPage.setObjectName("birdseyeConfigPage")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.birdseyeConfigPage)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_3 = QtWidgets.QFrame(self.birdseyeConfigPage)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_14.setSpacing(0)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.frame_10 = QtWidgets.QFrame(self.frame_3)
        self.frame_10.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_10.setStyleSheet("background-color: #44475a;")
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_10)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_2 = QtWidgets.QLabel(self.frame_10)
        self.label_2.setStyleSheet(
            "color: #ff79c6;\n" 'font: 16pt "MS Shell Dlg 2";'
        )
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(
            self.label_2, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_14.addWidget(self.frame_10)
        self.birdsEyeImageLabel_2 = QtWidgets.QFrame(self.frame_3)
        self.birdsEyeImageLabel_2.setMaximumSize(QtCore.QSize(16777215, 100))
        self.birdsEyeImageLabel_2.setStyleSheet(
            "QFrame#maskWeightesFrane {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}\n"
            ""
        )
        self.birdsEyeImageLabel_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.birdsEyeImageLabel_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.birdsEyeImageLabel_2.setObjectName("birdsEyeImageLabel_2")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(
            self.birdsEyeImageLabel_2
        )
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.birdsEyeLoadBtn = QtWidgets.QPushButton(self.birdsEyeImageLabel_2)
        self.birdsEyeLoadBtn.setMinimumSize(QtCore.QSize(0, 50))
        self.birdsEyeLoadBtn.setStyleSheet(
            "QPushButton#birdsEyeLoadBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    border: 1px solid #6272a4;\n"
            "    border-radius: 5px;\n"
            "    margin: 5px 15px;\n"
            "}\n"
            "\n"
            "QPushButton#birdsEyeLoadBtn:hover {\n"
            "    background-color: #6272a4;\n"
            "}\n"
            "\n"
            "QPushButton#birdsEyeLoadBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}\n"
            ""
        )
        self.birdsEyeLoadBtn.setObjectName("birdsEyeLoadBtn")
        self.horizontalLayout_9.addWidget(self.birdsEyeLoadBtn)
        self.birdsEyeImageLabel = QtWidgets.QLabel(self.birdsEyeImageLabel_2)
        self.birdsEyeImageLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #bd93f9;"
        )
        self.birdsEyeImageLabel.setObjectName("birdsEyeImageLabel")
        self.horizontalLayout_9.addWidget(
            self.birdsEyeImageLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.birdsEyeLoadLabel = QtWidgets.QLabel(self.birdsEyeImageLabel_2)
        self.birdsEyeLoadLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #bd93f9;"
        )
        self.birdsEyeLoadLabel.setObjectName("birdsEyeLoadLabel")
        self.horizontalLayout_9.addWidget(
            self.birdsEyeLoadLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_14.addWidget(self.birdsEyeImageLabel_2)
        self.birdsEyeOutputFrame = QtWidgets.QFrame(self.frame_3)
        self.birdsEyeOutputFrame.setMinimumSize(QtCore.QSize(416, 416))
        self.birdsEyeOutputFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.birdsEyeOutputFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.birdsEyeOutputFrame.setObjectName("birdsEyeOutputFrame")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.birdsEyeOutputFrame)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.birdsEyeOutputLabel = QtWidgets.QLabel(self.birdsEyeOutputFrame)
        self.birdsEyeOutputLabel.setText("")
        self.birdsEyeOutputLabel.setObjectName("birdsEyeOutputLabel")
        self.verticalLayout_7.addWidget(
            self.birdsEyeOutputLabel,
            0,
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
        )
        self.verticalLayout_14.addWidget(self.birdsEyeOutputFrame)
        self.frame_12 = QtWidgets.QFrame(self.frame_3)
        self.frame_12.setMaximumSize(QtCore.QSize(16777215, 150))
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout(self.frame_12)
        self.horizontalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_21.setSpacing(0)
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.birdseyeConfigPageBtn = QtWidgets.QPushButton(self.frame_12)
        self.birdseyeConfigPageBtn.setMaximumSize(QtCore.QSize(500, 16777215))
        self.birdseyeConfigPageBtn.setStyleSheet(
            "QPushButton#birdseyeConfigPageBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 16pt "MS Shell Dlg 2";\n'
            "    padding: 50px;\n"
            "    border: 1px solid #50fa7b;\n"
            "    border-radius: 5px\n"
            "}\n"
            "\n"
            "QPushButton#birdseyeConfigPageBtn:hover {\n"
            "    color: #50fa7b;\n"
            "}"
        )
        self.birdseyeConfigPageBtn.setObjectName("birdseyeConfigPageBtn")
        self.horizontalLayout_21.addWidget(self.birdseyeConfigPageBtn)
        self.verticalLayout_14.addWidget(self.frame_12)
        self.verticalLayout.addWidget(self.frame_3)
        self.stackedWidget.addWidget(self.birdseyeConfigPage)
        self.detectionPage = QtWidgets.QWidget()
        self.detectionPage.setObjectName("detectionPage")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.detectionPage)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.topBar = QtWidgets.QFrame(self.detectionPage)
        self.topBar.setMinimumSize(QtCore.QSize(0, 0))
        self.topBar.setMaximumSize(QtCore.QSize(16777215, 50))
        self.topBar.setStyleSheet(
            "background-color: #44475a;\n" "color: #ff79c6;"
        )
        self.topBar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.topBar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.topBar.setObjectName("topBar")
        self.horizontalLayout_48 = QtWidgets.QHBoxLayout(self.topBar)
        self.horizontalLayout_48.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_48.setSpacing(0)
        self.horizontalLayout_48.setObjectName("horizontalLayout_48")
        self.label_48 = QtWidgets.QLabel(self.topBar)
        self.label_48.setStyleSheet('font: 16pt "MS Shell Dlg 2";')
        self.label_48.setObjectName("label_48")
        self.horizontalLayout_48.addWidget(
            self.label_48, 0, QtCore.Qt.AlignHCenter
        )
        self.detectMaskButton = QtWidgets.QPushButton(self.topBar)
        self.detectMaskButton.setMinimumSize(QtCore.QSize(0, 50))
        self.detectMaskButton.setStyleSheet(
            "QPushButton#detectMaskButton {\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    border: 1px solid #ff79c6;\n"
            "    border-radius: 5px;\n"
            "    margin: 5px 15px;\n"
            "    color: #f8f8f2;\n"
            "}\n"
            "\n"
            "QPushButton#detectMaskButton:hover {\n"
            "    background-color: #ff79c6;\n"
            "}\n"
            "\n"
            "QPushButton#detectMaskButton:pressed {\n"
            "    color: #8be9fd;\n"
            "}\n"
            "\n"
            ""
        )
        self.detectMaskButton.setObjectName("detectMaskButton")
        self.horizontalLayout_48.addWidget(self.detectMaskButton)
        self.detectDistanceButton = QtWidgets.QPushButton(self.topBar)
        self.detectDistanceButton.setMinimumSize(QtCore.QSize(0, 50))
        self.detectDistanceButton.setStyleSheet(
            "QPushButton#detectDistanceButton {\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    border: 1px solid #ff79c6;\n"
            "    border-radius: 5px;\n"
            "    margin: 5px 15px;\n"
            "    color: #f8f8f2;\n"
            "    padding: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#detectDistanceButton:hover {\n"
            "    background-color: #ff79c6;\n"
            "}\n"
            "\n"
            "QPushButton#detectDistanceButton:pressed {\n"
            "    color: #8be9fd;\n"
            "}\n"
            "\n"
            "\n"
            ""
        )
        self.detectDistanceButton.setObjectName("detectDistanceButton")
        self.horizontalLayout_48.addWidget(self.detectDistanceButton)
        self.detectAllButton = QtWidgets.QPushButton(self.topBar)
        self.detectAllButton.setMinimumSize(QtCore.QSize(0, 50))
        self.detectAllButton.setStyleSheet(
            "QPushButton#detectAllButton {\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    border: 1px solid #ff79c6;\n"
            "    border-radius: 5px;\n"
            "    margin: 5px 15px;\n"
            "    color: #f8f8f2;\n"
            "    padding-left: 5px;\n"
            "    padding-right: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#detectAllButton:hover {\n"
            "    background-color: #ff79c6;\n"
            "}\n"
            "\n"
            "QPushButton#detectAllButton:pressed {\n"
            "    color: #8be9fd;\n"
            "}\n"
            "\n"
            "\n"
            ""
        )
        self.detectAllButton.setObjectName("detectAllButton")
        self.horizontalLayout_48.addWidget(self.detectAllButton)
        self.pauseDetectionBtn = QtWidgets.QPushButton(self.topBar)
        self.pauseDetectionBtn.setMinimumSize(QtCore.QSize(0, 50))
        self.pauseDetectionBtn.setStyleSheet(
            "QPushButton#pauseDetectionBtn {\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    border: 1px solid #ff79c6;\n"
            "    border-radius: 5px;\n"
            "    margin: 5px 15px;\n"
            "    color: #f8f8f2;\n"
            "    padding-left: 5px;\n"
            "    padding-right: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#pauseDetectionBtn:hover {\n"
            "    background-color: #ff79c6;\n"
            "}\n"
            "\n"
            "QPushButton#pauseDetectionBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}\n"
            "\n"
            "\n"
            ""
        )
        self.pauseDetectionBtn.setObjectName("pauseDetectionBtn")
        self.horizontalLayout_48.addWidget(
            self.pauseDetectionBtn, 0, QtCore.Qt.AlignVCenter
        )
        self.verticalLayout_2.addWidget(self.topBar)
        self.content = QtWidgets.QFrame(self.detectionPage)
        self.content.setMinimumSize(QtCore.QSize(0, 0))
        self.content.setStyleSheet("background-color: #282a36;")
        self.content.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.content.setFrameShadow(QtWidgets.QFrame.Raised)
        self.content.setObjectName("content")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout(self.content)
        self.horizontalLayout_19.setContentsMargins(0, 20, 10, 0)
        self.horizontalLayout_19.setSpacing(0)
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.sideBar = QtWidgets.QFrame(self.content)
        self.sideBar.setMaximumSize(QtCore.QSize(300, 16777215))
        self.sideBar.setStyleSheet(
            "background-color: #44475a;\n" "border-radius: 5px;"
        )
        self.sideBar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.sideBar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.sideBar.setObjectName("sideBar")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.sideBar)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.configFrame_4 = QtWidgets.QFrame(self.sideBar)
        self.configFrame_4.setMaximumSize(QtCore.QSize(16777215, 50))
        self.configFrame_4.setStyleSheet(
            "QFrame#configFrame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}\n"
            "\n"
            ""
        )
        self.configFrame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.configFrame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.configFrame_4.setObjectName("configFrame_4")
        self.horizontalLayout_38 = QtWidgets.QHBoxLayout(self.configFrame_4)
        self.horizontalLayout_38.setObjectName("horizontalLayout_38")
        self.label_37 = QtWidgets.QLabel(self.configFrame_4)
        self.label_37.setStyleSheet(
            'font: 16pt "MS Shell Dlg 2";\n' "color: #ffb86c;\n" ""
        )
        self.label_37.setObjectName("label_37")
        self.horizontalLayout_38.addWidget(
            self.label_37, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_11.addWidget(self.configFrame_4)
        self.config0Frame = QtWidgets.QFrame(self.sideBar)
        self.config0Frame.setMaximumSize(QtCore.QSize(16777215, 75))
        self.config0Frame.setStyleSheet(
            "QFrame#config0Frame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}"
        )
        self.config0Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.config0Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.config0Frame.setObjectName("config0Frame")
        self.horizontalLayout_49 = QtWidgets.QHBoxLayout(self.config0Frame)
        self.horizontalLayout_49.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_49.setSpacing(0)
        self.horizontalLayout_49.setObjectName("horizontalLayout_49")
        self.sourceBtn = QtWidgets.QPushButton(self.config0Frame)
        self.sourceBtn.setStyleSheet(
            "QPushButton#sourceBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 10pt "MS Shell Dlg 2";\n'
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    padding: 15px;\n"
            "    border: 1px solid #ffb86c;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "\n"
            "\n"
            "QPushButton#sourceBtn:hover {\n"
            "    background-color: #ffb86c;\n"
            "}\n"
            "\n"
            "QPushButton#sourceBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.sourceBtn.setObjectName("sourceBtn")
        self.horizontalLayout_49.addWidget(self.sourceBtn)
        self.sourceLabel = QtWidgets.QLabel(self.config0Frame)
        self.sourceLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #ffb86c;"
        )
        self.sourceLabel.setObjectName("sourceLabel")
        self.horizontalLayout_49.addWidget(
            self.sourceLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_11.addWidget(self.config0Frame)
        self.config7Frame = QtWidgets.QFrame(self.sideBar)
        self.config7Frame.setMaximumSize(QtCore.QSize(16777215, 75))
        self.config7Frame.setStyleSheet(
            "QFrame#config7Frame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}"
        )
        self.config7Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.config7Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.config7Frame.setObjectName("config7Frame")
        self.horizontalLayout_45 = QtWidgets.QHBoxLayout(self.config7Frame)
        self.horizontalLayout_45.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_45.setSpacing(0)
        self.horizontalLayout_45.setObjectName("horizontalLayout_45")
        self.camSourceBtn = QtWidgets.QPushButton(self.config7Frame)
        self.camSourceBtn.setStyleSheet(
            "QPushButton#camSourceBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    padding: 15px;\n"
            "    border: 1px solid #ffb86c;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#camSourceBtn:hover {\n"
            "    background-color: #ffb86c;\n"
            "}\n"
            "\n"
            "QPushButton#camSourceBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.camSourceBtn.setObjectName("camSourceBtn")
        self.horizontalLayout_45.addWidget(self.camSourceBtn)
        self.camSourceText = QtWidgets.QTextEdit(self.config7Frame)
        self.camSourceText.setMaximumSize(QtCore.QSize(16777215, 50))
        self.camSourceText.setStyleSheet(
            "margin-left: 5px;\n"
            "color: #ffb86c;\n"
            'font: 12pt "MS Shell Dlg 2";\n'
            "border: 0.5px solid #6272a4;\n"
            "border-radius: 5px;\n"
            "padding-top: 10px;"
        )
        self.camSourceText.setObjectName("camSourceText")
        self.horizontalLayout_45.addWidget(self.camSourceText)
        self.label_42 = QtWidgets.QLabel(self.config7Frame)
        self.label_42.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #ffb86c;"
        )
        self.label_42.setText("")
        self.label_42.setObjectName("label_42")
        self.horizontalLayout_45.addWidget(
            self.label_42, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_11.addWidget(self.config7Frame)
        self.config1Frame = QtWidgets.QFrame(self.sideBar)
        self.config1Frame.setMaximumSize(QtCore.QSize(16777215, 75))
        self.config1Frame.setStyleSheet(
            "QFrame#config1Frame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}"
        )
        self.config1Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.config1Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.config1Frame.setObjectName("config1Frame")
        self.horizontalLayout_39 = QtWidgets.QHBoxLayout(self.config1Frame)
        self.horizontalLayout_39.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_39.setSpacing(0)
        self.horizontalLayout_39.setObjectName("horizontalLayout_39")
        self.showMaskBtn = QtWidgets.QPushButton(self.config1Frame)
        self.showMaskBtn.setStyleSheet(
            "QPushButton#showMaskBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 10pt "MS Shell Dlg 2";\n'
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    padding: 15px;\n"
            "    border: 1px solid #ffb86c;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "\n"
            "\n"
            "QPushButton#showMaskBtn:hover {\n"
            "    background-color: #ffb86c;\n"
            "}\n"
            "\n"
            "QPushButton#showMaskBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.showMaskBtn.setObjectName("showMaskBtn")
        self.horizontalLayout_39.addWidget(self.showMaskBtn)
        self.showMaskLabel = QtWidgets.QLabel(self.config1Frame)
        self.showMaskLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #ffb86c;"
        )
        self.showMaskLabel.setObjectName("showMaskLabel")
        self.horizontalLayout_39.addWidget(
            self.showMaskLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_11.addWidget(self.config1Frame)
        self.config2Frame = QtWidgets.QFrame(self.sideBar)
        self.config2Frame.setMaximumSize(QtCore.QSize(16777215, 75))
        self.config2Frame.setStyleSheet(
            "QFrame#config2Frame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}"
        )
        self.config2Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.config2Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.config2Frame.setObjectName("config2Frame")
        self.horizontalLayout_40 = QtWidgets.QHBoxLayout(self.config2Frame)
        self.horizontalLayout_40.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_40.setSpacing(0)
        self.horizontalLayout_40.setObjectName("horizontalLayout_40")
        self.showScoreBtn = QtWidgets.QPushButton(self.config2Frame)
        self.showScoreBtn.setStyleSheet(
            "QPushButton#showScoreBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    padding: 15px;\n"
            "    border: 1px solid #ffb86c;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#showScoreBtn:hover {\n"
            "    background-color: #ffb86c;\n"
            "}\n"
            "\n"
            "QPushButton#showScoreBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.showScoreBtn.setObjectName("showScoreBtn")
        self.horizontalLayout_40.addWidget(self.showScoreBtn)
        self.showScoreLabel = QtWidgets.QLabel(self.config2Frame)
        self.showScoreLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #ffb86c;"
        )
        self.showScoreLabel.setObjectName("showScoreLabel")
        self.horizontalLayout_40.addWidget(
            self.showScoreLabel, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_11.addWidget(self.config2Frame)
        self.config3Frame = QtWidgets.QFrame(self.sideBar)
        self.config3Frame.setMaximumSize(QtCore.QSize(16777215, 75))
        self.config3Frame.setStyleSheet(
            "QFrame#config3Frame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}"
        )
        self.config3Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.config3Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.config3Frame.setObjectName("config3Frame")
        self.horizontalLayout_41 = QtWidgets.QHBoxLayout(self.config3Frame)
        self.horizontalLayout_41.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_41.setSpacing(0)
        self.horizontalLayout_41.setObjectName("horizontalLayout_41")
        self.scoreThreshBtn = QtWidgets.QPushButton(self.config3Frame)
        self.scoreThreshBtn.setStyleSheet(
            "QPushButton#scoreThreshBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    padding: 15px;\n"
            "    border: 1px solid #ffb86c;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#scoreThreshBtn:hover {\n"
            "    background-color: #ffb86c;\n"
            "}\n"
            "\n"
            "QPushButton#scoreThreshBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.scoreThreshBtn.setObjectName("scoreThreshBtn")
        self.horizontalLayout_41.addWidget(self.scoreThreshBtn)
        self.scoreThreshText = QtWidgets.QTextEdit(self.config3Frame)
        self.scoreThreshText.setMaximumSize(QtCore.QSize(16777215, 50))
        self.scoreThreshText.setStyleSheet(
            "margin-left: 5px;\n"
            "color: #ffb86c;\n"
            'font: 12pt "MS Shell Dlg 2";\n'
            "border: 0.5px solid #6272a4;\n"
            "border-radius: 5px;\n"
            "padding-top: 10px;"
        )
        self.scoreThreshText.setObjectName("scoreThreshText")
        self.horizontalLayout_41.addWidget(
            self.scoreThreshText,
            0,
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
        )
        self.verticalLayout_11.addWidget(self.config3Frame)
        self.config5Frame = QtWidgets.QFrame(self.sideBar)
        self.config5Frame.setMaximumSize(QtCore.QSize(16777215, 75))
        self.config5Frame.setStyleSheet(
            "QFrame#config5Frame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}"
        )
        self.config5Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.config5Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.config5Frame.setObjectName("config5Frame")
        self.horizontalLayout_42 = QtWidgets.QHBoxLayout(self.config5Frame)
        self.horizontalLayout_42.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_42.setSpacing(0)
        self.horizontalLayout_42.setObjectName("horizontalLayout_42")
        self.iouThreshBtn = QtWidgets.QPushButton(self.config5Frame)
        self.iouThreshBtn.setStyleSheet(
            "QPushButton#iouThreshBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    padding: 15px;\n"
            "    border: 1px solid #ffb86c;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#iouThreshBtn:hover {\n"
            "    background-color: #ffb86c;\n"
            "}\n"
            "\n"
            "QPushButton#iouThreshBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.iouThreshBtn.setObjectName("iouThreshBtn")
        self.horizontalLayout_42.addWidget(self.iouThreshBtn)
        self.iouThreshText = QtWidgets.QTextEdit(self.config5Frame)
        self.iouThreshText.setMaximumSize(QtCore.QSize(16777215, 50))
        self.iouThreshText.setStyleSheet(
            "margin-left: 5px;\n"
            "color: #ffb86c;\n"
            'font: 12pt "MS Shell Dlg 2";\n'
            "border: 0.5px solid #6272a4;\n"
            "border-radius: 5px;\n"
            "padding-top: 10px;margin-left: 5px;"
        )
        self.iouThreshText.setObjectName("iouThreshText")
        self.horizontalLayout_42.addWidget(self.iouThreshText)
        self.verticalLayout_11.addWidget(self.config5Frame)
        self.config6Frame = QtWidgets.QFrame(self.sideBar)
        self.config6Frame.setMaximumSize(QtCore.QSize(16777215, 75))
        self.config6Frame.setStyleSheet(
            "QFrame#config6Frame {\n"
            "    border-bottom: 1px solid #f8f8f2;\n"
            "}"
        )
        self.config6Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.config6Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.config6Frame.setObjectName("config6Frame")
        self.horizontalLayout_43 = QtWidgets.QHBoxLayout(self.config6Frame)
        self.horizontalLayout_43.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_43.setSpacing(0)
        self.horizontalLayout_43.setObjectName("horizontalLayout_43")
        self.socialDistanceBtn = QtWidgets.QPushButton(self.config6Frame)
        self.socialDistanceBtn.setStyleSheet(
            "QPushButton#socialDistanceBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    padding: 15px;\n"
            "    border: 1px solid #ffb86c;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#socialDistanceBtn:hover {\n"
            "    background-color: #ffb86c;\n"
            "}\n"
            "\n"
            "QPushButton#socialDistanceBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.socialDistanceBtn.setObjectName("socialDistanceBtn")
        self.horizontalLayout_43.addWidget(self.socialDistanceBtn)
        self.socialDistanceText = QtWidgets.QTextEdit(self.config6Frame)
        self.socialDistanceText.setMaximumSize(QtCore.QSize(16777215, 50))
        self.socialDistanceText.setStyleSheet(
            "margin-left: 5px;\n"
            "color: #ffb86c;\n"
            'font: 12pt "MS Shell Dlg 2";\n'
            "border: 0.5px solid #6272a4;\n"
            "border-radius: 5px;\n"
            "padding-top: 10px;"
        )
        self.socialDistanceText.setObjectName("socialDistanceText")
        self.horizontalLayout_43.addWidget(self.socialDistanceText)
        self.verticalLayout_11.addWidget(self.config6Frame)
        self.frame_18 = QtWidgets.QFrame(self.sideBar)
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.frame_18.setMaximumSize(QtCore.QSize(16777215, 175))
        self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.frame_18)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.configureBirdsEye = QtWidgets.QPushButton(self.frame_18)
        self.configureBirdsEye.setStyleSheet(
            "QPushButton#configureBirdsEye {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    padding: 15px;\n"
            "    border: 1px solid #ffb86c;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#configureBirdsEye:hover {\n"
            "    background-color: #ffb86c;\n"
            "}\n"
            "\n"
            "QPushButton#configureBirdsEye:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.configureBirdsEye.setObjectName("configureBirdsEye")
        self.verticalLayout_18.addWidget(self.configureBirdsEye)
        self.configureRefObj = QtWidgets.QPushButton(self.frame_18)
        self.configureRefObj.setStyleSheet(
            "QPushButton#configureRefObj {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    padding: 15px;\n"
            "    border: 1px solid #ffb86c;\n"
            "    border-radius: 5px;\n"
            "}\n"
            "\n"
            "QPushButton#configureRefObj:hover {\n"
            "    background-color: #ffb86c;\n"
            "}\n"
            "\n"
            "QPushButton#configureRefObj:pressed {\n"
            "    color: #8be9fd;\n"
            "}"
        )
        self.configureRefObj.setObjectName("configureRefObj")
        self.verticalLayout_18.addWidget(self.configureRefObj)
        self.verticalLayout_11.addWidget(self.frame_18)
        self.horizontalLayout_19.addWidget(self.sideBar)
        self.mainContent = QtWidgets.QFrame(self.content)
        self.mainContent.setStyleSheet("")
        self.mainContent.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainContent.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mainContent.setLineWidth(0)
        self.mainContent.setObjectName("mainContent")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.mainContent)
        self.verticalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_12.setSpacing(0)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.infoBar = QtWidgets.QFrame(self.mainContent)
        self.infoBar.setMaximumSize(QtCore.QSize(16777215, 190))
        self.infoBar.setStyleSheet(
            "background-color: #44475a;\n"
            "margin-left: 5px;\n"
            "color: #bd93f9;\n"
            "border-radius: 5px;"
        )
        self.infoBar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.infoBar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.infoBar.setObjectName("infoBar")
        self.horizontalLayout_46 = QtWidgets.QHBoxLayout(self.infoBar)
        self.horizontalLayout_46.setObjectName("horizontalLayout_46")
        self.frame_5 = QtWidgets.QFrame(self.infoBar)
        self.frame_5.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.sourcePathBtn = QtWidgets.QPushButton(self.frame_6)
        self.sourcePathBtn.setMaximumSize(QtCore.QSize(150, 16777215))
        self.sourcePathBtn.setStyleSheet(
            "QPushButton#sourcePathBtn {\n"
            "    color: #f8f8f2;\n"
            '    font: 12pt "MS Shell Dlg 2";\n'
            "    border: 1px solid #6272a4;\n"
            "    border-radius: 5px;\n"
            "    margin: 5px 15px;\n"
            "}\n"
            "\n"
            "QPushButton#sourcePathBtn:hover {\n"
            "    background-color: #6272a4;\n"
            "}\n"
            "\n"
            "QPushButton#sourcePathBtn:pressed {\n"
            "    color: #8be9fd;\n"
            "}\n"
            ""
        )
        self.sourcePathBtn.setObjectName("sourcePathBtn")
        self.horizontalLayout_6.addWidget(self.sourcePathBtn)
        self.addSourceLabel = QtWidgets.QLabel(self.frame_6)
        self.addSourceLabel.setStyleSheet('font: 10pt "MS Shell Dlg 2";')
        self.addSourceLabel.setText("")
        self.addSourceLabel.setObjectName("addSourceLabel")
        self.horizontalLayout_6.addWidget(self.addSourceLabel)
        self.verticalLayout_6.addWidget(self.frame_6)
        self.frame_8 = QtWidgets.QFrame(self.frame_5)
        self.frame_8.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_47 = QtWidgets.QHBoxLayout(self.frame_8)
        self.horizontalLayout_47.setObjectName("horizontalLayout_47")
        self.frame_4 = QtWidgets.QFrame(self.frame_8)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.socialDistanceLabel = QtWidgets.QLabel(self.frame_4)
        self.socialDistanceLabel.setStyleSheet('font: 12pt "MS Shell Dlg 2";')
        self.socialDistanceLabel.setObjectName("socialDistanceLabel")
        self.verticalLayout_5.addWidget(self.socialDistanceLabel)
        self.fpsLabel = QtWidgets.QLabel(self.frame_4)
        self.fpsLabel.setStyleSheet('font: 12pt "MS Shell Dlg 2";')
        self.fpsLabel.setObjectName("fpsLabel")
        self.verticalLayout_5.addWidget(self.fpsLabel)
        self.horizontalLayout_47.addWidget(self.frame_4)
        self.frame_9 = QtWidgets.QFrame(self.frame_8)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.maskCountLabel = QtWidgets.QLabel(self.frame_9)
        self.maskCountLabel.setStyleSheet('font: 12pt "MS Shell Dlg 2";')
        self.maskCountLabel.setObjectName("maskCountLabel")
        self.verticalLayout_13.addWidget(self.maskCountLabel)
        self.nomaskCountLabel = QtWidgets.QLabel(self.frame_9)
        self.nomaskCountLabel.setStyleSheet('font: 12pt "MS Shell Dlg 2";')
        self.nomaskCountLabel.setObjectName("nomaskCountLabel")
        self.verticalLayout_13.addWidget(self.nomaskCountLabel)
        self.distanceViolationLabel = QtWidgets.QLabel(self.frame_9)
        self.distanceViolationLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";'
        )
        self.distanceViolationLabel.setObjectName("distanceViolationLabel")
        self.verticalLayout_13.addWidget(self.distanceViolationLabel)
        self.horizontalLayout_47.addWidget(self.frame_9)
        self.verticalLayout_6.addWidget(self.frame_8)
        self.horizontalLayout_46.addWidget(self.frame_5)
        self.verticalLayout_12.addWidget(self.infoBar)
        self.outputFrame = QtWidgets.QFrame(self.mainContent)
        self.outputFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.outputFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.outputFrame.setObjectName("outputFrame")
        # self.outputFrame.setMinimumSize(QtCore.QSize(416, 416))
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.outputFrame)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.detectionOutputlabel = QtWidgets.QLabel(self.outputFrame)
        self.detectionOutputlabel.setMaximumSize(
            QtCore.QSize(16777215, 16777215)
        )
        self.detectionOutputlabel.setText("")
        self.detectionOutputlabel.setObjectName("detectionOutputlabel")
        self.horizontalLayout_7.addWidget(
            self.detectionOutputlabel, 0, QtCore.Qt.AlignHCenter
        )
        self.verticalLayout_12.addWidget(self.outputFrame)
        self.horizontalLayout_19.addWidget(self.mainContent)
        self.verticalLayout_2.addWidget(self.content)
        self.stackedWidget.addWidget(self.detectionPage)
        self.horizontalLayout_20.addWidget(self.stackedWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # general buttons clicks

        self.refObjLoadText.textChanged.connect(
            lambda: update_ref_obj_dim(self)
        )

        # page buttons

        self.yoloButton.clicked.connect(lambda: yolo_btn_clicked(self))
        self.yoloTinyButton.clicked.connect(
            lambda: yolo_tiny_btn_clicked(self)
        )
        self.loadWeightsPageBtn.clicked.connect(
            lambda: load_weights_page_btn_clicked(self)
        )
        self.refObjConfigPageBtn.clicked.connect(
            lambda: ref_obj_page_btn(self)
        )
        self.birdseyeConfigPageBtn.clicked.connect(
            lambda: birdseye_page_btn_clicked(self)
        )

        # load buttons

        self.maskWeightsBtn.clicked.connect(lambda: load_mask_weights(self))
        self.distanceWeightsBtn.clicked.connect(
            lambda: load_distance_weights(self)
        )
        self.sourcePathBtn.clicked.connect(lambda: load_detection_source(self))
        
        # config buttons

        self.sourceBtn.clicked.connect(lambda: source_btn_clicked(self))
        self.camSourceBtn.clicked.connect(lambda: cam_source_clicked(self))
        self.showMaskBtn.clicked.connect(lambda: show_mask_clicked(self))
        self.showScoreBtn.clicked.connect(lambda: show_score_clicked(self))
        self.scoreThreshBtn.clicked.connect(lambda: score_thresh_clicked(self))
        self.iouThreshBtn.clicked.connect(lambda: iou_thresh_clicked(self))
        self.socialDistanceBtn.clicked.connect(
            lambda: social_distance_clicked(self)
        )
        self.configureBirdsEye.clicked.connect(
            lambda: configure_birds_eye(self)
        )
        self.configureRefObj.clicked.connect(lambda: configure_ref_obj(self))

        # detection buttons

        self.detectMaskButton.clicked.connect(lambda: detect_mask(self))
        self.detectDistanceButton.clicked.connect(
            lambda: detect_distance(self)
        )
        self.detectAllButton.clicked.connect(lambda: detect_all(self))
        self.pauseDetectionBtn.clicked.connect(lambda: pause_detection(self))

        # bird's-eye view config

        self.birdsEyeLoadBtn.clicked.connect(
            lambda: load_birds_eye_config_source(self)
        )
        self.birdsEyeOutputLabel.mouseDoubleClickEvent = (
            lambda event: draw_point(self, event)
        )

        # reference object config

        self.referenceObjectLoadButton.clicked.connect(
            lambda: load_ref_obj_config_src(self)
        )

        self.refObjOutputLabel.mouseDoubleClickEvent = (
            lambda event: draw_ref_point(self, event)
        )

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(
            _translate("MainWindow", "Coivd-19 Warning System")
        )
        self.label_4.setText(_translate("MainWindow", "Detection Algorithm"))
        self.yoloButton.setText(_translate("MainWindow", "YOLOv4"))
        self.yoloTinyButton.setText(_translate("MainWindow", "YOLOv4 Tiny"))
        self.label.setText(_translate("MainWindow", "Load Weights"))
        self.maskWeightsBtn.setText(
            _translate("MainWindow", "Choose Mask Detection Weights")
        )
        self.maskPathLabel.setText(_translate("MainWindow", "Path: "))
        self.maskLoadedLabel.setText(
            _translate("MainWindow", "Weights not loaded")
        )
        self.distanceWeightsBtn.setText(
            _translate(
                "MainWindow", "Choose Social Distance  Detection Weights"
            )
        )
        self.distancePathLabel.setText(_translate("MainWindow", "Path: "))
        self.distanceLoadedLabel.setText(
            _translate("MainWindow", "Weights not loaded")
        )
        self.loadWeightsPageBtn.setText(_translate("MainWindow", "Next"))
        self.label_3.setText(
            _translate("MainWindow", "Configure Reference Object")
        )
        self.referenceObjectLoadButton.setText(
            _translate("MainWindow", "Load Reference Object Source")
        )
        self.refObjLoadText.setHtml(
            _translate(
                "MainWindow",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                '<p align="center" style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">1.8</p></body></html>',
            )
        )
        self.refObjLoadLabel.setText(
            _translate("MainWindow", "Reference object not configured")
        )
        self.refObjConfigPageBtn.setText(_translate("MainWindow", "Next"))
        self.label_2.setText(
            _translate("MainWindow", "Configure Bird's-Eye View")
        )
        self.birdsEyeLoadBtn.setText(
            _translate("MainWindow", "Choose Bird's-Eye configuration image")
        )
        self.birdsEyeImageLabel.setText(_translate("MainWindow", "Path: "))
        self.birdsEyeLoadLabel.setText(
            _translate("MainWindow", "Bird's-Eye View not configured")
        )
        self.birdseyeConfigPageBtn.setText(_translate("MainWindow", "Next"))
        self.label_48.setText(_translate("MainWindow", "Detect:"))
        self.detectMaskButton.setText(_translate("MainWindow", "Mask "))
        self.detectDistanceButton.setText(
            _translate("MainWindow", "Social distance Violation")
        )
        self.detectAllButton.setText(
            _translate("MainWindow", "Full Detection")
        )
        self.pauseDetectionBtn.setText(_translate("MainWindow", "Pause"))
        self.label_37.setText(_translate("MainWindow", "Configuration"))
        self.sourceBtn.setText(_translate("MainWindow", "Change Source"))
        self.sourceLabel.setText(_translate("MainWindow", "Video"))
        self.camSourceBtn.setText(_translate("MainWindow", "Cam Source"))
        self.camSourceText.setHtml(
            _translate(
                "MainWindow",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                '<p align="center" style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">0</p></body></html>',
            )
        )
        self.showMaskBtn.setText(_translate("MainWindow", "Show Mask"))
        self.showMaskLabel.setText(_translate("MainWindow", "True"))
        self.showScoreBtn.setText(_translate("MainWindow", "Show Score"))
        self.showScoreLabel.setText(_translate("MainWindow", "True"))
        self.scoreThreshBtn.setText(
            _translate("MainWindow", "Score Threshold")
        )
        self.scoreThreshText.setHtml(
            _translate(
                "MainWindow",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                '<p align="center" style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">0.3</p></body></html>',
            )
        )
        self.iouThreshBtn.setText(_translate("MainWindow", "IOU Threshold"))
        self.iouThreshText.setHtml(
            _translate(
                "MainWindow",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                '<p align="center" style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">0.45</p></body></html>',
            )
        )
        self.socialDistanceBtn.setText(
            _translate("MainWindow", "Social Distance")
        )
        self.socialDistanceText.setHtml(
            _translate(
                "MainWindow",
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                '<p align="center" style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">1.5</p></body></html>',
            )
        )
        self.configureBirdsEye.setText(
            _translate("MainWindow", "Configure Bird's-Eye View")
        )
        self.configureRefObj.setText(
            _translate("MainWindow", "Configure Reference Object")
        )
        self.sourcePathBtn.setText(_translate("MainWindow", "Add source"))
        self.socialDistanceLabel.setText(
            _translate("MainWindow", "Social Distance: 1.5 meters")
        )
        self.fpsLabel.setText(_translate("MainWindow", "FPS: 0"))
        self.maskCountLabel.setText(_translate("MainWindow", "Mask: 0"))
        self.nomaskCountLabel.setText(_translate("MainWindow", "No mask: 0"))
        self.distanceViolationLabel.setText(
            _translate("MainWindow", "Social distance violations: 0")
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
