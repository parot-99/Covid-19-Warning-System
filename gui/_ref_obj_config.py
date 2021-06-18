import cv2
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage


def reload_obj_ref_img(self):
    display_frame = QImage(
        self.ref_obj_src,
        416,
        416,
        self.ref_obj_src.strides[0],
        QImage.Format_RGB888,
    )

    self.refObjOutputLabel.setPixmap(QtGui.QPixmap.fromImage(display_frame))


def load_ref_obj_config_src(self):
    source_path = QFileDialog.getOpenFileName()[0]
    self.ref_obj_counter = 0
    self.ref_obj_circles = np.zeros((2, 2), np.int)

    if source_path == "":
        return

    ref_obj_src = cv2.imread(source_path)
    ref_obj_src = cv2.cvtColor(ref_obj_src, cv2.COLOR_BGR2RGB)
    self.ref_obj_src = cv2.resize(ref_obj_src, (416, 416))
    self.refObjOutputLabel.setMinimumSize(QtCore.QSize(416, 416))
    self.refObjOutputLabel.resize(416, 416)
    reload_obj_ref_img(self)


def draw_ref_point(self, event):
    if self.ref_obj_counter == 2 or self.ref_obj_counter > 2:
        return

    x = event.pos().x()
    y = event.pos().y()
    self.ref_obj_circles[self.ref_obj_counter] = x, y
    self.ref_obj_counter += 1

    cv2.circle(
        self.ref_obj_src,
        (x, y),
        3,
        (0, 255, 0),
        2,
    )

    if self.ref_obj_counter == 2:
        top = self.ref_obj_circles[0][1]
        bottom = self.ref_obj_circles[1][1]
        self.pixel_to_meter_scale = (bottom - top) / self.ref_obj_dim
        self.min_social_distance = (
            self.social_distance * self.pixel_to_meter_scale
        )
        self.refObjLoadLabel.setText("Reference object configured")
        self.refObjLoadLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #50fa7b;"
        )

    reload_obj_ref_img(self)
