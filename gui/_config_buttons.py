def source_btn_clicked(self):
    if self.source == self.source_list[0]:
        self.source = self.source_list[1]

    elif self.source == self.source_list[1]:
        self.source = self.source_list[2]

    elif self.source == self.source_list[2]:
        self.source = self.source_list[0]

    self.sourceLabel.setText(self.source)


def show_mask_clicked(self):
    self.show_mask = not self.show_mask
    self.showMaskLabel.setText(f"{str(self.show_mask)}")


def show_score_clicked(self, *args, **kwargs):
    self.show_score = not self.show_score
    self.showScoreLabel.setText(f"{str(self.show_score)}")


def score_thresh_clicked(self):
    new_threshod = self.scoreThreshText.toPlainText()
    self.score_threshold = float(new_threshod)


def iou_thresh_clicked(self):
    new_threshod = self.iouThreshText.toPlainText()
    self.iou_threshold = float(new_threshod)


def cam_source_clicked(self):
    new_cam_source = self.camSourceText.toPlainText()
    self.cam_source = float(new_cam_source)


def social_distance_clicked(self):
    new_social_distance = self.socialDistanceText.toPlainText()
    self.social_distance = float(new_social_distance)
    self.min_social_distance = self.social_distance * self.pixel_to_meter_scale
    self.socialDistanceLabel.setText(
        f"Social Distance: {self.social_distance} meters"
    )


def configure_birds_eye(self):
    self.stackedWidget.setCurrentIndex(3)


def configure_ref_obj(self):
    self.stackedWidget.setCurrentIndex(2)
