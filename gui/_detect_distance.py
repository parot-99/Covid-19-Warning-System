import cv2
import time
import tensorflow as tf
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QImage
from scipy.spatial import distance as dist
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def process_distance_frame(self, src, class_names):
    frame = src.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    width = self.outputFrame.frameGeometry().width() - 50
    height = self.outputFrame.frameGeometry().height() - 50
    frame = cv2.resize(frame, (width, height))

    image_data = src.copy()
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(image_data, (416, 416))
    image_data = image_data / 255.0
    image_data = np.expand_dims(image_data, axis=0).astype("float32")
    image_data = tf.constant(image_data)

    prediction = self.distance_detector(image_data)

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
        max_total_size=1000,
        iou_threshold=self.iou_threshold,
        score_threshold=self.score_threshold,
    )

    bounding_boxes = [
        boxes.numpy(),
        scores.numpy(),
        classes.numpy(),
        valid_detections.numpy(),
    ]

    violations_count = 0
    classes_count = len(class_names)
    frame_height, frame_width, _ = frame.shape
    boxes, scores, classes, box_count = bounding_boxes
    poly = Polygon(
        np.array(
            [
                self.circles[0],
                self.circles[1],
                self.circles[3],
                self.circles[2],
            ],
            np.int32,
        )
    )

    violate = list()
    centroids = list()

    for i, box in enumerate(boxes[0]):
        if i > box_count:
            break

        w = (box[3] - box[1]) / 2
        h = (box[2] - box[0]) / 2
        centroid = (
            (box[1] + w) * 416,
            (box[0] + h) * 416,
        )
        centroids.append(centroid)

    transformed_centroids = np.float32(centroids).reshape(-1, 1, 2)
    transformed_centroids = cv2.perspectiveTransform(
        transformed_centroids, self.matrix
    )
    transformed_centroids = transformed_centroids.reshape(-1, 2)

    centroids = list(centroids)
    centroids = centroids[0 : box_count[0]]
    transformed_centroids = list(transformed_centroids)
    transformed_centroids = transformed_centroids[0 : box_count[0]]

    if len(transformed_centroids) > 2:
        D = dist.cdist(
            transformed_centroids,
            transformed_centroids,
            metric="euclidean",
        )

        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < self.min_social_distance:
                    point1 = Point(centroids[i])
                    point2 = Point(centroids[j])

                    if poly.contains(point1) and poly.contains(point2):
                        violate.append(i)
                        violate.append(j)

    for i in range(box_count[0]):
        class_id = int(classes[0][i])
        score = scores[0][i]

        point = Point(centroids[i])

        if not poly.contains(point):
            continue

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

        if i in violate:
            color = (255, 0, 0)
            violations_count += 1

        else:
            color = (0, 0, 255)

        score = int(score * 100)

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
                f"Person [{score}%]",
                (bounding_box["left"], bounding_box["top"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        cx, cy = centroids[i]
        cx = int(cx / 416 * frame_width)
        cy = int(cy / 416 * frame_height)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), 1)

    poly_pts = np.array(
        [
            [
                int(self.circles[0][0] / 416 * frame_width),
                int(self.circles[0][1] / 416 * frame_height),
            ],
            [
                int(self.circles[1][0] / 416 * frame_width),
                int(self.circles[1][1] / 416 * frame_height),
            ],
            [
                int(self.circles[3][0] / 416 * frame_width),
                int(self.circles[3][1] / 416 * frame_height),
            ],
            [
                int(self.circles[2][0] / 416 * frame_width),
                int(self.circles[2][1] / 416 * frame_height),
            ]
        ],
        np.int32,
    )
    poly_pts = poly_pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [poly_pts], True, (0, 255, 255))
    display_frame = QImage(
        frame, width, height, frame.strides[0], QImage.Format_RGB888
    )

    self.detectionOutputlabel.setPixmap(QtGui.QPixmap.fromImage(display_frame))
    self.distanceViolationLabel.setText(
        f"Social Distance Violations: {violations_count}"
    )


def detect_distance(self):
    self.maskCountLabel.setText(f"Mask: {0}")
    self.nomaskCountLabel.setText(f"No Mask: {0}")
    self.distanceViolationLabel.setText(f"Social Distance Violations: {0}")
    
    class_names = {0: "person"}
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

            self.process_distance_frame(self, src, class_names)

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

        self.process_distance_frame(self, frame, class_names)
