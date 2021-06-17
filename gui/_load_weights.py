from PyQt5.QtWidgets import QFileDialog
from yolo.model.yolo import Yolo


def load_mask_weights(self):
    model = (
        Yolo(2)
        if self.detection_model == "yolo"
        else Yolo(2, tiny=True)
    )
    mask_weights_path = QFileDialog.getOpenFileName()[0]

    if mask_weights_path == "":
        return

    self.maskPathLabel.setText(f"Path: {mask_weights_path}")
    self.maskLoadedLabel.setText("Loading weights...")
    model.load_weights(mask_weights_path)
    self.mask_detector = model.get_graph()
    self.maskLoadedLabel.setText("Weights loaded!")
    self.maskLoadedLabel.setStyleSheet(
        'font: 12pt "MS Shell Dlg 2";\n' "color: #50fa7b;"
    )


def load_distance_weights(self):
    model = (
        Yolo(80)
        if self.detection_model == "yolo"
        else Yolo(80, tiny=True)
    )
    distance_weights_path = QFileDialog.getOpenFileName()[0]

    if distance_weights_path == "":
        return

    self.distancePathLabel.setText(f"Path: {distance_weights_path}")
    self.distanceLoadedLabel.setText("Loading weights...")
    model.load_weights(distance_weights_path)
    self.distance_detector = model.get_graph()
    self.distanceLoadedLabel.setText("Weights loaded!")
    self.distanceLoadedLabel.setStyleSheet(
        'font: 12pt "MS Shell Dlg 2";\n' "color: #50fa7b;"
    )