def yolo_btn_clicked(self):
    self.detection_model = 'yolo'
    self.stackedWidget.setCurrentIndex(1)

def yolo_tiny_btn_clicked(self):
    self.detection_model = 'yolol-tiny'
    self.stackedWidget.setCurrentIndex(1)

def load_weights_page_btn_clicked(self):
    self.stackedWidget.setCurrentIndex(4)

def birdseye_page_btn_clicked(self):
    self.stackedWidget.setCurrentIndex(4)

def ref_obj_page_btn(self):
    self.stackedWidget.setCurrentIndex(4)
