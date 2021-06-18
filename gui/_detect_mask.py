import cv2
import time
import tensorflow as tf
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QImage


def process_mask_frame(self, src, class_names):
    frame = src.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    width = self.outputFrame.frameGeometry().width() - 50
    height = self.outputFrame.frameGeometry().height() - 50
    frame = cv2.resize(frame, (width, height))

    image_data = src.copy()
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(image_data, (416, 416))
    image_data = image_data / 255.0
    image_data = image_data.astype("float32")
    image_data = np.expand_dims(image_data, axis=0).astype("float32")
    image_data = tf.constant(image_data)

    prediction = self.mask_detector(image_data)
    boxes = prediction[0, :, 0:4]
    pred_conf = prediction[0, :, 4:]
    boxes = np.reshape(boxes, (1, boxes.shape[0], boxes.shape[1]))
    pred_conf = np.reshape(
        pred_conf, (1, pred_conf.shape[0], pred_conf.shape[1])
    )

    (
        boxes,
        scores,
        classes,
        valid_detections,
    ) = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf,
            (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1]),
        ),
        max_output_size_per_class=1000,
        max_total_size=2000,
        iou_threshold=self.iou_threshold,
        score_threshold=self.score_threshold,
    )

    bounding_boxes = [
        boxes.numpy(),
        scores.numpy(),
        classes.numpy(),
        valid_detections.numpy(),
    ]
    mask_count = 0
    no_mask_count = 0
    classes_count = len(class_names)
    frame_height, frame_width, _ = frame.shape
    boxes, scores, classes, box_count = bounding_boxes

    for i in range(box_count[0]):
        class_id = int(classes[0][i])
        score = scores[0][i]

        if int(class_id < 0) or int(class_id > classes_count):
            continue

        box = boxes[0][i]
        top = box[0] * frame_height
        bottom = box[2] * frame_height
        left = box[1] * frame_width
        right = box[3] * frame_width

        bounding_box = {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
        }

        if class_id == 0:
            label = "no-mask"
            no_mask_count += 1

        if class_id == 1:
            label = "mask"
            mask_count += 1

        color = (255, 0, 0) if label == "no-mask" else (0, 255, 0)
        score = int(score * 100)

        if label == "mask" and not self.show_mask:
            continue

        cv2.rectangle(
            frame,
            (bounding_box["left"], bounding_box["top"]),
            (bounding_box["right"], bounding_box["bottom"]),
            color,
            1,
        )

        if self.show_score:
            cv2.putText(
                frame,
                f"{label} [{score}%]",
                (bounding_box["left"], bounding_box["top"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    display_frame = QImage(
        frame, width, height, frame.strides[0], QImage.Format_RGB888
    )

    self.detectionOutputlabel.setPixmap(QtGui.QPixmap.fromImage(display_frame))

    self.maskCountLabel.setText(f"Mask: {mask_count}")
    self.nomaskCountLabel.setText(f"No Mask: {no_mask_count}")


def detect_mask(self):
    self.maskCountLabel.setText(f"Mask: {0}")
    self.nomaskCountLabel.setText(f"No Mask: {0}")
    self.distanceViolationLabel.setText(f"Social Distance Violations: {0}")
    
    class_names = {0: "no-mask", 1: "mask"}
    print(f"[INFO]: Running inference from {self.source}")

    if self.source == "Video" or self.source == "Cam":
        if self.source == "Video" and self.source_path == "":
            return

        detection_source = self.source_path

        if self.source == "Cam":
            detection_source = self.cam_source

        self.detecting = 1
        cap = cv2.VideoCapture(detection_source, cv2.CAP_ANY)
        avg_fps = []

        while True:
            if not self.detecting:
                cap.release()
                cv2.destroyAllWindows()
                break

            prev_time = time.time()
            retval, src = cap.read(0)

            if not retval:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            self.process_mask_frame(self, src, class_names)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            fps = int(1 / (time.time() - prev_time))
            self.fpsLabel.setText(f"FPS: {fps}")
            avg_fps.append(fps)

        cap.release()
        cv2.destroyAllWindows()

        if len(avg_fps) != 0:
            avg_fps = sum(avg_fps) / len(avg_fps)
            print(f"[INFO]: Average FPS: {avg_fps}")

    elif self.source == "Image":
        frame = cv2.imread(self.source_path)

        if frame is None:
            print("\n[WARNING]: Please enter a valid image path")
            return

        self.process_mask_frame(self, frame, class_names)
