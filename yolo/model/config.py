from easydict import EasyDict as edict
import numpy as np

cfg = edict()

anchors = np.array(
    [
        12, 16, 19, 36, 40, 28,
        36, 75, 76, 55, 72, 146,
        142, 110, 192, 243, 459, 401,
    ]
) 
anchors = anchors.reshape(3, 3, 2)

anchors_tiny = np.array(
    [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]
) 
anchors_tiny = anchors_tiny.reshape(2, 3, 2)

cfg.INPUT_SIZE = 416
cfg.ANCHORS = anchors
cfg.ANCHORS_TINY = anchors_tiny
cfg.STRIDES = np.array([8, 16, 32])
cfg.STRIDES_TINY = np.array([16, 32])
cfg.XSCALE = np.array([1.2, 1.1, 1.05])
cfg.XSCALE_TINY = np.array([1.05, 1.05])
cfg.ANCHOR_PER_SCALE = 3
cfg.IOU_LOSS_THRESH = 0.45

# layers

cfg.CONV_LAYERS = [
    1, 2, 6, 3, 4, 5, 8, 9, 14, 10, 11, 12, 13, 16, 17, 28, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 30, 31, 42, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
    41, 44, 45, 52, 46, 47, 48, 49, 50, 51, 54, 55, 56, 57, 59, 60, 61 ,62,
    63, 66, 67, 68, 69, 70, 71, 72, 75, 76, 77, 78, 79, 94, 97, 80, 82, 83, 
    84, 85, 86, 95, 98, 87, 89, 90, 91, 92, 93, 96, 99
]
cfg.RESIDUAL_LAYERS = [
    4, 11, 12, 19, 20, 21, 22, 23, 24, 25, 26, 33, 
    34, 35, 36, 37, 38, 39, 40, 47, 48,49, 50
]
cfg.OUTPUT_LAYERS = [93,101,109]
cfg.CONV_LAYERS_TINY = [
    1, 2, 3, 5, 6, 8, 11, 13, 
    14, 16, 19, 21, 22, 24, 27, 
    28, 33, 35, 29, 32, 34
]
cfg.RESIDUAL_LAYERS_TINY = []
cfg.OUTPUT_LAYERS_TINY = [17,20]
