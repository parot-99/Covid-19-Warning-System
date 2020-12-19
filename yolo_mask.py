import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class YoloMask:
    def __init__(self, detector_path):
        self.__class_names = {0: 'no-mask', 1: 'mask'}
        self.__show_masks = False
        self.__show_fps = False
        self.__show_scores = False
        self.__write_detection = False
        self.__score_threshold = 0.3
        self.__iou_threshold = 0.45
        loaded_detector = load_model(detector_path, compile=False)
        self.mask_detector = loaded_detector.signatures['serving_default']

    def detect_from_image(self, image_path):
        frame = cv2.imread(image_path)

        if frame is None:
           print('\n[WARNING]: Please enter a valid image path')
           return

        self.__detect_frame(frame)

        if self.__write_detection:
            cv2.imwrite('results.jpg', frame)

        cv2.imshow('Image', frame)
        cv2.waitKey(0)

    def detect_from_video(self, src=0):
        cap = cv2.VideoCapture(src, cv2.CAP_ANY)

        if self.__write_detection:
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            output = cv2.VideoWriter(
                filename='output.avi',
                apiPreference=cv2.CAP_ANY,
                fourcc=cv2.VideoWriter_fourcc('M','J','P','G'),
                fps=fps,
                frameSize=(width, height)
            )

        try:
            while cap.isOpened():
                prev_time = time.time()
                retval, frame = cap.read(0)
                self.__detect_frame(frame)
                cv2.imshow('Frame', frame)

                if self.__write_detection:
                    output.write(frame)

                if self.__show_fps:
                    fps = int(1/(time.time() - prev_time))
                    print("FPS: {}".format(fps))

                if cv2.waitKey(1) & 0xff == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def __detect_frame(self, frame):
        image_data = frame.copy()
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(image_data, (416, 416))
        image_data = image_data / 255.
        image_data = image_data.astype('float32')
        image_data = np.expand_dims(image_data, axis=0)
        image_data = tf.constant(image_data)

        prediction = self.mask_detector(image_data)

        for key, value in prediction.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = \
            tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf,
                    (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
                ),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=self.__iou_threshold,
                score_threshold=self.__score_threshold
            )

        bounding_boxes = [
            boxes.numpy(),
            scores.numpy(),
            classes.numpy(),
            valid_detections.numpy()
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
                'left': int(left),
                'top': int(top),
                'right': int(right),
                'bottom': int(bottom)
            }


            if class_id == 0:
                label = 'no-mask'
                no_mask_count += 1

            if class_id == 1:
                label = 'mask'
                mask_count += 1

            self.__draw_bounding_box(frame, label, bounding_box, score)

        self.__display_info(frame, box_count, mask_count, no_mask_count)


    def __draw_bounding_box(self, frame, label, bounding_box, score):
        color = (0, 0, 255) if label == 'no-mask' else (0, 255, 0)
        score = int(score * 100)

        if label == 'mask' and not self.__show_masks:
            return

        cv2.rectangle(
            frame,
            (bounding_box['left'], bounding_box['top']),
            (bounding_box['right'], bounding_box['bottom']),
            color,
            1
        )

        if self.__show_scores:
            cv2.putText(
                frame,
                f'{label} [{score}%]',
                (bounding_box['left'], bounding_box['top'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )


    def __display_info(self, frame, box_count, mask_count, no_mask_count):
        cv2.putText(
            frame,
            f'Mask: {mask_count}',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f'No-mask: {no_mask_count}',
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    @property
    def show_masks(self):
        return self.__show_masks

    @property
    def show_fps(self):
        return self.__show_fps

    @property
    def show_scores(self):
        return self.__show_scores

    @property
    def write_detection(self):
        return self.__write_detection

    @property
    def score_threshold(self):
        return self.__score_threshold

    @property
    def iou_threshold(self):
        return self.__iou_threshold

    @show_masks.setter
    def show_masks(self, show_masks):
        self.__show_masks = show_masks

    @show_fps.setter
    def show_fps(self, show_fps):
        self.__show_fps = show_fps

    @show_scores.setter
    def show_scores(self, show_scores):
        self.__show_scores = show_scores

    @write_detection.setter
    def write_detection(self, write_detection):
        self.__write_detection = write_detection

    @score_threshold.setter
    def score_threshold(self, score_threshold):
        self.__score_threshold = score_threshold

    @iou_threshold.setter
    def iou_threshold(self, iou_threshold):
        self.__iou_threshold = iou_threshold

