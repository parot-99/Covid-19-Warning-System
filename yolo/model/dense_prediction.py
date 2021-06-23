import tensorflow as tf
from .config import cfg


class YoloHead(tf.keras.layers.Layer):
    def __init__(self, grid_size, classes, strides, anchors, xyscale, i):
        super().__init__()
        self.grid_size = grid_size
        self.classes = classes
        self.strides = strides
        self.anchors = anchors
        self.xyscale = xyscale
        self.i = i

    def call(self, feature_map):
        batch_size = tf.shape(feature_map)[0]
        conv_output = tf.reshape(
            feature_map,
            (batch_size, self.grid_size, self.grid_size, 3, 5 + self.classes),
        )

        bbox_xy, bbox_wh, detection_conf, classes_prob = tf.split(
            conv_output, (2, 2, 1, self.classes), axis=-1
        )

        xy_grid = tf.meshgrid(
            tf.range(self.grid_size), tf.range(self.grid_size)
        )
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)
        xy_grid = tf.tile(
            tf.expand_dims(xy_grid, axis=0),
            [batch_size, 1, 1, 3, 1],
        )
        xy_grid = tf.cast(xy_grid, tf.float32)

        bbox_xy_sigmoid = tf.sigmoid(bbox_xy)
        detection_conf_sigmoid = tf.sigmoid(detection_conf)
        classes_prob_sigmoid = tf.sigmoid(classes_prob)

        prediction_xy = (
            (bbox_xy_sigmoid * self.xyscale[self.i])
            - 0.5 * (self.xyscale[self.i] - 1)
            + xy_grid
        ) * self.strides[self.i]
        prediction_wh = tf.exp(bbox_wh) * self.anchors[self.i]

        prediction_xywh = tf.concat([prediction_xy, prediction_wh], axis=-1)
        prediction_prob = detection_conf_sigmoid * classes_prob_sigmoid

        prediction_xywh = tf.reshape(prediction_xywh, (batch_size, -1, 4))
        prediction_prob = tf.reshape(
            prediction_prob, (batch_size, -1, self.classes)
        )

        return prediction_xywh, prediction_prob


class FilterLayer(tf.keras.layers.Layer):
    def __init__(self, input_size, score_threshold=0.4):
        super().__init__()
        self.input_size = input_size
        self.score_threshold = score_threshold

    def call(self, bounding_boxes, scores):
        input_size = self.input_size
        score_threshold = self.score_threshold
        bounding_boxes = tf.concat(bounding_boxes, axis=1)
        scores = tf.concat(scores, axis=1)
        scores_max = tf.math.reduce_max(scores, axis=-1)

        mask = scores_max >= score_threshold
        class_boxes = tf.boolean_mask(bounding_boxes, mask)
        pred_conf = tf.boolean_mask(scores, mask)
        class_boxes = tf.reshape(
            class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]]
        )
        pred_conf = tf.reshape(
            pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]]
        )

        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

        input_size = tf.cast(input_size, dtype=tf.float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw / 2.0)) / input_size
        box_maxes = (box_yx + (box_hw / 2.0)) / input_size
        boxes = tf.concat(
            [
                box_mins[..., 0:1],
                box_mins[..., 1:2],
                box_maxes[..., 0:1],
                box_maxes[..., 1:2],
            ],
            axis=-1,
        )

        predictions = tf.concat([boxes, pred_conf], axis=-1)

        return predictions


def dense_prediction(feature_maps, classes, tiny=False):
    bbox_tensors = []
    prob_tensors = []

    if tiny:
        yolo_head_1 = YoloHead(
            cfg.INPUT_SIZE // 16,
            classes,
            cfg.STRIDES_TINY,
            cfg.ANCHORS_TINY,
            cfg.XYSCALE_TINY,
            0,
        )(feature_maps[0])

        bbox_tensors.append(yolo_head_1[0])
        prob_tensors.append(yolo_head_1[1])

        yolo_head_2 = YoloHead(
            cfg.INPUT_SIZE // 32,
            classes,
            cfg.STRIDES_TINY,
            cfg.ANCHORS_TINY,
            cfg.XYSCALE_TINY,
            1,
        )(feature_maps[1])

        bbox_tensors.append(yolo_head_2[0])
        prob_tensors.append(yolo_head_2[1])

    else:
        yolo_head_1 = YoloHead(
            cfg.INPUT_SIZE // 8,
            classes,
            cfg.STRIDES,
            cfg.ANCHORS,
            cfg.XYSCALE,
            0,
        )(feature_maps[0])

        bbox_tensors.append(yolo_head_1[0])
        prob_tensors.append(yolo_head_1[1])

        yolo_head_2 = YoloHead(
            cfg.INPUT_SIZE // 16,
            classes,
            cfg.STRIDES,
            cfg.ANCHORS,
            cfg.XYSCALE,
            1,
        )(feature_maps[1])

        bbox_tensors.append(yolo_head_2[0])
        prob_tensors.append(yolo_head_2[1])

        yolo_head_3 = YoloHead(
            cfg.INPUT_SIZE // 32,
            classes,
            cfg.STRIDES,
            cfg.ANCHORS,
            cfg.XYSCALE,
            2,
        )(feature_maps[2])

        bbox_tensors.append(yolo_head_3[0])
        prob_tensors.append(yolo_head_3[1])

    predictions = FilterLayer(
        input_size=tf.constant([cfg.INPUT_SIZE, cfg.INPUT_SIZE]),
        score_threshold=0.2
    )(bbox_tensors, prob_tensors)

    return predictions
