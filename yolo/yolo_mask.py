import time
import cv2
import numpy as np
import tensorflow as tf
import json
from .model.yolo import Yolo
from imutils.video import FileVideoStream


with open("config.json", "r") as file:
    config = json.load(file)


class YoloMask:
    def __init__(self, tiny=False):
        self.__class_names = {0: "no-mask", 1: "mask"}
        self.__show_masks = config["showMasks"]
        self.__show_fps = config["showFPS"]
        self.__show_scores = config["showScores"]
        self.__score_threshold = config["scoreThreshold"]
        self.__iou_threshold = config["iouThreshold"]
        self.__write_detection = config["writeDetection"]
        print("[INFO]: Building model")
        model = Yolo(2, tiny)
        print("[INFO]: Model built succesfully")
        print("[INFO]: Loading detector weights")
        model.load_weights(config['weightsPath'])
        print("[INFO]: Detector weights loaded")
        self.mask_detector = model.get_graph()

    def detect_from_image(self, image_path):
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
        cap = cv2.VideoCapture(src, cv2.CAP_ANY)
        # fvs = FileVideoStream(src).start()
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
            retval, frame = cap.read(0)
            prev_time = time.time()
            # frame = fvs.read()

            if not retval:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            self.__detect_frame(frame)

            if not config["dontShow"]:
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.__write_detection:
                output.write(frame)

            if self.__show_fps:
                fps = int(1 / (time.time() - prev_time))
                avg_fps.append(fps)
                print("[INFO]: FPS: {}".format(fps))

        cap.release()
        cv2.destroyAllWindows()

        if len(avg_fps) != 0:
            avg_fps = sum(avg_fps) / len(avg_fps)
            print(f"[INFO]: Average FPS: {avg_fps}")

    
    def get_bounding_boxes(self, image_path):
        frame = cv2.imread(image_path)

        if frame is None:
            print("\n[WARNING]: Please enter a valid image path")
            return

        image_data = frame.copy()
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(image_data, (416, 416))
        image_data = image_data / 255.0
        image_data = image_data.astype("float32")
        image_data = np.expand_dims(image_data, axis=0)
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
            iou_threshold=self.__iou_threshold,
            score_threshold=self.__score_threshold,
        )

        bounding_boxes = [
            boxes.numpy(),
            scores.numpy(),
            classes.numpy(),
            valid_detections.numpy(),
        ]

        return bounding_boxes

    def __detect_frame(self, frame):
        image_data = frame.copy()
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(image_data, (416, 416))
        image_data = image_data / 255.0
        image_data = image_data.astype("float32")
        image_data = np.expand_dims(image_data, axis=0)
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
        mask_count = 0
        no_mask_count = 0
        classes_count = len(self.__class_names)
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

            self.__draw_bounding_box(frame, label, bounding_box, score)

        self.__display_info(frame, box_count, mask_count, no_mask_count)

    def __draw_bounding_box(self, frame, label, bounding_box, score):
        color = (0, 0, 255) if label == "no-mask" else (0, 255, 0)
        score = int(score * 100)

        if label == "mask" and not self.__show_masks:
            return

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
                f"{label} [{score}%]",
                (bounding_box["left"], bounding_box["top"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def __display_info(self, frame, box_count, mask_count, no_mask_count):
        cv2.putText(
            frame,
            f"Mask: {mask_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
        )

        cv2.putText(
            frame,
            f"No-mask: {no_mask_count}",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
    

