import time
import cv2
import numpy as np
import json
import tensorflow as tf
from scipy.spatial import distance as dist
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from .model.yolo import Yolo


with open("config.json", "r") as file:
    config = json.load(file)


class YoloSocialDistance:
    def __init__(self, tiny=False):
        self.__class_names = {0: "person"}
        self.__show_masks = config["showMasks"]
        self.__show_fps = config["showFPS"]
        self.__show_scores = config["showScores"]
        self.__score_threshold = config["scoreThreshold"]
        self.__iou_threshold = config["iouThreshold"]
        self.__write_detection = config["writeDetection"]
        self.__min_distance = 0
        print("[INFO]: Building model")
        model = Yolo(80, tiny)
        print("[INFO]: Model built succesfully")
        print("[INFO]: Loading detector weights")
        model.load_weights(config["weightsPath"])
        print("[INFO]: Detector weights loaded")
        self.pedestrian_detector = model.get_graph()

    def detect_from_image(self, image_path):
        self.__pixel_to_meter()
        self.__get_perspecive_points()
        frame = cv2.imread(image_path)

        if frame is None:
            print("\n[WARNING]: Please enter a valid image path")
            return

        self.__detect_frame(frame)

        if self.__write_detection:
            cv2.imwrite("prediction.jpg", frame)

        if not config["dontShow"]:
            cv2.imshow("Image", frame)
            key = cv2.waitKey(0)

            if key == ord("q"):
                cv2.destroyAllWindows()

    def detect_from_video(self, src=0):
        self.__pixel_to_meter()
        self.__get_perspecive_points()
        cap = cv2.VideoCapture(src, cv2.CAP_ANY)
        avg_fps = []

        if self.__write_detection:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            output = cv2.VideoWriter(
                filename="prediction.avi",
                apiPreference=cv2.CAP_ANY,
                fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                fps=fps,
                frameSize=(width, height),
            )

        while True:
            prev_time = time.time()
            retval, frame = cap.read(0)

            if not retval:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            self.__detect_frame(frame)
            pts = np.array(
                [
                    self.__circles[0],
                    self.__circles[1],
                    self.__circles[3],
                    self.__circles[2],
                ],
                np.int32,
            )
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 255))

            if not config["dontShow"]:
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.__show_fps:
                fps = int(1 / (time.time() - prev_time))
                avg_fps.append(fps)
                print("FPS: {}".format(fps))

            if self.__write_detection:
                output.write(frame)

        cap.release()
        cv2.destroyAllWindows()

        if len(avg_fps) != 0:
            avg_fps = sum(avg_fps) / len(avg_fps)
            print(f"[INFO]: Average FPS: {avg_fps}")

    def __detect_frame(self, frame):
        image_data = frame.copy()
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(image_data, (416, 416))
        image_data = image_data / 255.0
        image_data = image_data.astype("float32")
        image_data = np.expand_dims(image_data, axis=0)
        image_data = tf.constant(image_data)

        prediction = self.pedestrian_detector(image_data)

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
            max_output_size_per_class=2000,
            max_total_size=2000,
            iou_threshold=self.__iou_threshold,
            score_threshold=self.__score_threshold,
        )

        bounding_boxes = [
            boxes.numpy(),
            scores.numpy(),
            classes.numpy(),
            valid_detections.numpy(),
        ]

        self.__process_frame(frame, bounding_boxes)

    def __process_frame(self, frame, bounding_boxes):
        violations_count = 0
        classes_count = len(self.__class_names)
        frame_height, frame_width, _ = frame.shape
        boxes, scores, classes, box_count = bounding_boxes
        poly = Polygon(
            np.array(
                [
                    self.__circles[0],
                    self.__circles[1],
                    self.__circles[3],
                    self.__circles[2],
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
                (box[1] + w) * frame_width,
                (box[0] + h) * frame_height,
            )
            centroids.append(centroid)

        transformed_centroids = np.float32(centroids).reshape(-1, 1, 2)
        transformed_centroids = cv2.perspectiveTransform(
            transformed_centroids, self.__matrix
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
                    if D[i, j] < self.__min_distance:
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
                color = (0, 0, 255)
                violations_count += 1

            else:
                color = (255, 0, 0)

            self.__draw_bounding_box(
                frame, bounding_box, score, centroids, color, i
            )

        self.__display_info(frame, violate, violations_count)

    def __draw_bounding_box(
        self, frame, bounding_box, score, centroids, color, i
    ):
        score = int(score * 100)

        cv2.rectangle(
            frame,
            (bounding_box["left"], bounding_box["top"]),
            (bounding_box["right"], bounding_box["bottom"]),
            color,
            1,
        )

        if self.__show_scores:
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
        cx = int(cx)
        cy = int(cy)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), 1)

    def __display_info(self, frame, violate, violations_count):
        text = "Social Distancing Violations: {}".format(violations_count)
        cv2.putText(
            frame,
            text,
            (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3,
        )

    def __pixel_to_meter(self):
        if config["dontShowMinDistance"] == 0:
            img = cv2.imread(config["socialDistanceFrame"])
            drawing_img = img.copy()

            self.__reference_obj_points = np.zeros((2, 2), np.int)
            self.__counter = 0

            def draw_rectangle(event, x, y, flags, params):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.__reference_obj_points[self.__counter] = x, y
                    cv2.circle(drawing_img, (x, y), 3, (0, 255, 0), 2)
                    self.__counter += 1

            cv2.namedWindow("Choose bounding points for reference object", 0)
            cv2.setMouseCallback(
                "Choose bounding points for reference object", draw_rectangle
            )

            while True:
                cv2.imshow(
                    "Choose bounding points for reference object", drawing_img
                )
                key = cv2.waitKey(1)

                if self.__counter == 2:
                    top = self.__reference_obj_points[0][1]
                    bottom = self.__reference_obj_points[1][1]
                    scale = (bottom - top) / config["pedestrianHeight"]
                    self.__min_distance = scale * config["socialDistance"]
                    print(
                        f"[INFO]: Min social distance: {self.__min_distance}"
                    )
                    cv2.destroyAllWindows()
                    break

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    break

        else:
            self.__min_distance = config["dontShowMinDistance"]
            

    def __get_perspecive_points(self):
        img = cv2.imread(config["socialDistanceFrame"])
        drawing_img = img.copy()

        if len(config["birdEyeCoordinates"][0]) == 0:
            self.__circles = np.zeros((4, 2), np.int)
            self.__counter = 0

            def draw_rectangle(event, x, y, flags, params):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.__circles[self.__counter] = x, y
                    cv2.circle(drawing_img, (x, y), 3, (0, 255, 0), 2)
                    self.__counter += 1

            cv2.namedWindow("Choose birds's-eye view 4 corners", 0)
            cv2.setMouseCallback(
                "Choose birds's-eye view 4 corners", draw_rectangle
            )

            while True:
                cv2.imshow("Choose birds's-eye view 4 corners", drawing_img)
                key = cv2.waitKey(1)

                if self.__counter == 4:
                    max_width = max(self.__circles[1][0], self.__circles[3][0])
                    min_width = min(self.__circles[0][0], self.__circles[2][0])
                    width = max_width - min_width
                    max_height = max(
                        self.__circles[2][1], self.__circles[3][1]
                    )
                    min_height = min(
                        self.__circles[0][1], self.__circles[1][1]
                    )
                    height = max_height - min_height
                    pts1 = np.float32(
                        [
                            self.__circles[0],
                            self.__circles[1],
                            self.__circles[2],
                            self.__circles[3],
                        ]
                    )
                    pts2 = np.float32(
                        [[0, 0], [width, 0], [0, height], [width, height]]
                    )

                    self.__matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    warped = cv2.warpPerspective(
                        drawing_img, self.__matrix, (width, height)
                    )
                    pts = np.array(
                        [
                            self.__circles[0],
                            self.__circles[1],
                            self.__circles[3],
                            self.__circles[2],
                        ],
                        np.int32,
                    )
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(drawing_img, [pts], True, (0, 255, 255))
                    cv2.destroyAllWindows()
                    break

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    break

            print("[INFO]: Bird's eye view corner points:")
            print(f"[INFO]: Top left corner: {self.__circles[0]}")
            print(f"[INFO]: Top right corner: {self.__circles[1]}")
            print(f"[INFO]: Bottom left corner: {self.__circles[2]}")
            print(f"[INFO]: Bottom right corner: {self.__circles[3]}")

        else:
            self.__circles = config["birdEyeCoordinates"]
            max_width = max(self.__circles[1][0], self.__circles[3][0])
            min_width = min(self.__circles[0][0], self.__circles[2][0])
            width = max_width - min_width
            max_height = max(self.__circles[2][1], self.__circles[3][1])
            min_height = min(self.__circles[0][1], self.__circles[1][1])
            height = max_height - min_height
            pts1 = np.float32(
                [
                    self.__circles[0],
                    self.__circles[1],
                    self.__circles[2],
                    self.__circles[3],
                ]
            )
            pts2 = np.float32(
                [[0, 0], [width, 0], [0, height], [width, height]]
            )

            self.__matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(
                drawing_img, self.__matrix, (width, height)
            )
            pts = np.array(
                [
                    self.__circles[0],
                    self.__circles[1],
                    self.__circles[3],
                    self.__circles[2],
                ],
                np.int32,
            )
