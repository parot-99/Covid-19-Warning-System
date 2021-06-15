import tensorflow as tf
from .config import cfg


def decode(conv_output, grid_size, classes, strides, anchors, xyscale, i):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.keras.backend.reshape(
        conv_output, (batch_size, grid_size, grid_size, 3, 5 + classes)
    )


    bbox_xy, bbox_wh, detection_conf, classes_prob = tf.split(
        conv_output, (2, 2, 1, classes), axis=-1
    )

    xy_grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)
    xy_grid = tf.tile(
        tf.keras.backend.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1]
    )
    xy_grid = tf.cast(xy_grid, tf.float32)

    bbox_xy_sigmoid = tf.sigmoid(bbox_xy)
    detection_conf_sigmoid = tf.sigmoid(detection_conf)
    classes_prob_sigmoid = tf.sigmoid(classes_prob)
    
    prediction_xy = (
        (bbox_xy_sigmoid * xyscale[i])
        - 0.5 * (xyscale[i] - 1)
        + xy_grid
    ) * strides[i]
    prediction_wh = tf.exp(bbox_wh) * anchors[i]
     
    prediction_xywh = tf.concat([prediction_xy, prediction_wh], axis=-1)
    prediction_prob = detection_conf_sigmoid * classes_prob_sigmoid

    prediction_xywh = tf.reshape(prediction_xywh, (batch_size, -1, 4))
    prediction_prob = tf.reshape(prediction_prob, (batch_size, -1, classes))


    return prediction_xywh, prediction_prob


def filter_boxes(box_xywh, scores, input_shape, score_threshold=0.4):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(
        class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]]
    )
    pred_conf = tf.reshape(
        pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]]
    )

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.0)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.0)) / input_shape
    boxes = tf.concat(
        [
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2],
        ],
        axis=-1,
    )

    return (boxes, pred_conf)


def dense_prediction(feature_maps, classes, tiny=False):
    bbox_tensors = []
    prob_tensors = []

    if not tiny:
        for i, feature_map in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(
                    feature_map,
                    cfg.INPUT_SIZE // 8,
                    classes,
                    cfg.STRIDES,
                    cfg.ANCHORS,
                    cfg.XYSCALE,
                    i,
                )
        
            elif i == 1:
                output_tensors = decode(
                    feature_map,
                    cfg.INPUT_SIZE // 16,
                    classes,
                    cfg.STRIDES,
                    cfg.ANCHORS,
                    cfg.XYSCALE,
                    i,
                )
        
            else:
                output_tensors = decode(
                    feature_map,
                    cfg.INPUT_SIZE // 32,
                    classes,
                    cfg.STRIDES,
                    cfg.ANCHORS,
                    cfg.XYSCALE,
                    i,
                )
        
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])

    else:
        for i, feature_map in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(
                    feature_map,
                    cfg.INPUT_SIZE // 16,
                    classes,
                    cfg.STRIDES_TINY,
                    cfg.ANCHORS_TINY,
                    cfg.XYSCALE_TINY,
                    i,
                )
                
            else:
                output_tensors = decode(
                    feature_map,
                    cfg.INPUT_SIZE // 32,
                    classes,
                    cfg.STRIDES_TINY,
                    cfg.ANCHORS_TINY,
                    cfg.XYSCALE_TINY,
                    i,
                )
            
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    boxes, pred_conf = filter_boxes(
        pred_bbox,
        pred_prob,
        score_threshold=0.2,
        input_shape=tf.constant([cfg.INPUT_SIZE, cfg.INPUT_SIZE]),
    )
    pred = tf.concat([boxes, pred_conf], axis=-1)

    return pred