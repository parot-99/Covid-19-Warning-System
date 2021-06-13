import tensorflow as tf
from .layers import Conv, Upsample
from .config import cfg


def yolo_neck(cspdarknet_layers, classes):
    rote_1, route_2, layers = cspdarknet_layers
    # block 8

    route = layers
    layers = Conv(256, 1, activation="leaky")(layers)
    layers = Upsample(2)(layers)
    route_2 = Conv(256, 1)(route_2)
    layers = tf.keras.layers.Concatenate(axis=-1)([route_2, layers])

    # block 9

    layers = Conv(256, 1, activation="leaky")(layers)
    layers = Conv(512, 3, activation="leaky")(layers)
    layers = Conv(256, 1, activation="leaky")(layers)
    layers = Conv(512, 3, activation="leaky")(layers)
    layers = Conv(256, 1, activation="leaky")(layers)

    # block 10

    route_2 = layers
    layers = Conv(128, 1, activation="leaky")(layers)
    layers = Upsample(2)(layers)
    route_1 = Conv(128, 1, activation="leaky")(route_1)
    layers = tf.keras.layers.Concatenate(axis=-1)([route_1, layers])

    # block 11

    layers = Conv(128, 1, activation="leaky")(layers)
    layers = Conv(256, 3, activation="leaky")(layers)
    layers = Conv(128, 1, activation="leaky")(layers)
    layers = Conv(256, 3, activation="leaky")(layers)
    layers = Conv(128, 1, activation="leaky")(layers)

    # small_bbox block

    route_1 = layers
    layers = Conv(256, 3, activation="leaky")(layers)
    small_bbox = Conv(3 * (classes + 5), 1, activation="linear", bn=False)(
        layers
    )

    # block 12

    layers = Conv(256, 3, downsample=True)(route_1)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route_2])

    # block 13

    layers = Conv(256, 1, activation="leaky")(layers)
    layers = Conv(512, 3, activation="leaky")(layers)
    layers = Conv(256, 1, activation="leaky")(layers)
    layers = Conv(512, 3, activation="leaky")(layers)
    layers = Conv(256, 1, activation="leaky")(layers)

    # medium_bbox block

    route_2 = layers
    layers = Conv(512, 3, activation="leaky")(layers)
    medium_bbox = Conv(
        3 * (classes + 5), 1, activation="linear", bn=False
    )(layers)

    # block 14

    layers = Conv(512, 3, downsample=True)(route_2)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route])

    # block 15

    layers = Conv(512, 1, activation="leaky")(layers)
    layers = Conv(1024, 3, activation="leaky")(layers)
    layers = Conv(512, 1, activation="leaky")(layers)
    layers = Conv(1024, 3, activation="leaky")(layers)
    layers = Conv(512, 1, activation="leaky")(layers)

    # large block

    layers = Conv(1024, 3, activation="leaky")(layers)
    large_bbox = Conv(3 * (classes + 5), 1, activation="linear", bn=False)(
        layers
    )

    feature_maps = (small_bbox, medium_bbox, large_bbox)

    return feature_maps


def yolo_tiny_neck(cspdarknet_tiny_layers, classes):
    route_1, layers = cspdarknet_tiny_layers

    layers = Conv(256, 1, activation="leaky")(layers)

    route = Conv(512, 3, activation="leaky")(layers)
    large_bbox = Conv(3 * (classes + 5), 1, activation="linear", bn=False)(
        route
    )

    layers = Conv(128, 1, activation="leaky")(layers)
    layers = Upsample(2)(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route_1])

    route = Conv(256, 3, activation="leaky")(layers)
    medium_bbox = Conv(3 * (classes + 5), 1, activation="linear", bn=False)(
        route
    )

    feature_maps = (medium_bbox, large_bbox)

    return feature_maps
