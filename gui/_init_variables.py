import numpy as np
from gui._detect_mask import process_mask_frame
from gui._detect_distance import process_distance_frame
from gui._detect_all import process_all_frame

def init_variables(self):
    self.source_list = ["Video", "Cam", "Image"]
    self.source = "Video"
    self.source_path = ""
    self.cam_source = 0
    self.show_mask = True
    self.show_score = True
    self.score_threshold = 0.3
    self.iou_threshold = 0.45
    self.social_distance = 1.5
    self.ref_obj_dim = 1.8
    self.pixel_to_meter_scale = 0
    self.min_social_distance = 0
    self.birds_eye_counter = 0
    self.circles = np.zeros((4, 2), np.int)
    self.dynamic_circles = np.zeros((4, 2), np.int)
    self.ref_obj_counter = 0
    self.ref_obj_circles = np.zeros((2, 2), np.int)
    self.detection_model = 'yolo'
    self.process_mask_frame = process_mask_frame
    self.process_distance_frame = process_distance_frame
    self.process_all_frame = process_all_frame
    self.detecting = 0