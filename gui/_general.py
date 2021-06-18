from PyQt5.QtWidgets import QFileDialog


def load_detection_source(self):
    self.source_path = QFileDialog.getOpenFileName()[0]
    self.addSourceLabel.setText(f"Path: {self.source_path}")

def pause_detection(self):
    self.detecting = 0
            
def update_ref_obj_dim(self):
    self.ref_obj_dim = float(self.refObjLoadText.toPlainText())
