import tensorflow as tf
from .config import cfg


def decode(conv_output, output_size, classes, strides, anchors, xscale, i):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.keras.backend.reshape(
        conv_output, (batch_size, output_size, output_size, 3, 5 + classes)
    )

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (2, 2, 1, classes), axis=-1
    )

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.keras.backend.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)
    xy_grid = tf.keras.backend.tile(
        tf.keras.backend.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1]
    )

    xy_grid = tf.keras.backend.cast(xy_grid, tf.float32)

    pred_xy = (
        (tf.keras.backend.sigmoid(conv_raw_dxdy) * xscale[i])
        - 0.5 * (xscale[i] - 1)
        + xy_grid
    ) * strides[i]
    pred_wh = tf.keras.backend.exp(conv_raw_dwdh) * anchors[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.keras.backend.reshape(pred_prob, (batch_size, -1, classes))
    pred_xywh = tf.keras.backend.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob


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
                    cfg.XSCALE,
                    i,
                )
        
            elif i == 1:
                output_tensors = decode(
                    feature_map,
                    cfg.INPUT_SIZE // 16,
                    classes,
                    cfg.STRIDES,
                    cfg.ANCHORS,
                    cfg.XSCALE,
                    i,
                )
        
            else:
                output_tensors = decode(
                    feature_map,
                    cfg.INPUT_SIZE // 32,
                    classes,
                    cfg.STRIDES,
                    cfg.ANCHORS,
                    cfg.XSCALE,
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
                    cfg.XSCALE_TINY,
                    i,
                )
                
            else:
                output_tensors = decode(
                    feature_map,
                    cfg.INPUT_SIZE // 32,
                    classes,
                    cfg.STRIDES_TINY,
                    cfg.ANCHORS_TINY,
                    cfg.XSCALE_TINY,
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