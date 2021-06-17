import cv2
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage


def reload_birds_view_image(self):
    display_frame = QImage(
        self.birds_eye_src,
        416,
        416,
        self.birds_eye_src.strides[0],
        QImage.Format_RGB888,
    )

    self.birdsEyeOutputLabel.setPixmap(QtGui.QPixmap.fromImage(display_frame))


def load_birds_eye_config_source(self):
    source_path = QFileDialog.getOpenFileName()[0]
    self.birds_eye_counter = 0
    self.birdsEyeImageLabel.setText(f"Path: {source_path}")

    if source_path == "":
        return

    self.birds_eye_src = cv2.imread(source_path)
    self.birds_eye_src = cv2.cvtColor(self.birds_eye_src, cv2.COLOR_BGR2RGB)
    self.birds_eye_src = cv2.resize(self.birds_eye_src, (416, 416))
    self.birdsEyeOutputLabel.setMinimumSize(QtCore.QSize(416, 416))
    self.birdsEyeOutputLabel.resize(416, 416)
    reload_birds_view_image(self)


def draw_point(self, event):
    if self.birds_eye_counter == 4 or self.birds_eye_counter > 4:
        return

    x = event.pos().x()
    y = event.pos().y()
    self.circles[self.birds_eye_counter] = x, y
    self.birds_eye_counter += 1

    cv2.circle(
        self.birds_eye_src,
        (x, y),
        3,
        (0, 255, 0),
        2,
    )

    if self.birds_eye_counter == 4:
        max_width = max(self.circles[1][0], self.circles[3][0])
        min_width = min(self.circles[0][0], self.circles[2][0])
        width = max_width - min_width
        max_height = max(self.circles[2][1], self.circles[3][1])
        min_height = min(self.circles[0][1], self.circles[1][1])
        height = max_height - min_height
        pts1 = np.float32(
            [
                self.circles[0],
                self.circles[1],
                self.circles[2],
                self.circles[3],
            ]
        )
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(
            self.birds_eye_src, self.matrix, (width, height)
        )
        poly_pts = np.array(
            [
                self.circles[0],
                self.circles[1],
                self.circles[3],
                self.circles[2],
            ],
            np.int32,
        )
        poly_pts = poly_pts.reshape((-1, 1, 2))
        cv2.polylines(self.birds_eye_src, [poly_pts], True, (0, 255, 255))
        self.birdsEyeLoadLabel.setText('Bird\'s-Eye View configured')
        self.birdsEyeLoadLabel.setStyleSheet(
            'font: 12pt "MS Shell Dlg 2";\n' "color: #50fa7b;"
        )
    reload_birds_view_image(self)
