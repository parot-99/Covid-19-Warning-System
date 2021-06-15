import tensorflow as tf
from .layers import Conv, Residual, SSP, route_group


def cspdarknet53(input_layer):

    # block 1

    layers = Conv(32, 3)(input_layer)
    layers = Conv(64, 3, downsample=True)(layers)
    route = layers
    route = Conv(64, 1)(route)
    layers = Conv(64, 1)(layers)
    layers = Residual((32, 64), (1, 3))(layers)
    layers = Conv(64, 1)(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route])

    # block 2

    layers = Conv(64, 1)(layers)
    layers = Conv(128, 3, downsample=True)(layers)
    route = layers
    route = Conv(64, 1)(route)
    layers = Conv(64, 1)(layers)
    layers = Residual((64, 64), (1, 3))(layers)
    layers = Residual((64, 64), (1, 3))(layers)
    layers = Conv(64, 1)(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route])

    # block 3

    layers = Conv(128, 1)(layers)
    layers = Conv(256, 3, downsample=True)(layers)
    route = layers
    route = Conv(128, 1)(route)
    layers = Conv(128, 1)(layers)
    layers = Residual((128, 128), (1, 3))(layers)
    layers = Residual((128, 128), (1, 3))(layers)
    layers = Residual((128, 128), (1, 3))(layers)
    layers = Residual((128, 128), (1, 3))(layers)
    layers = Residual((128, 128), (1, 3))(layers)
    layers = Residual((128, 128), (1, 3))(layers)
    layers = Residual((128, 128), (1, 3))(layers)
    layers = Residual((128, 128), (1, 3))(layers)
    layers = Conv(128, 1)(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route])

    # block 4

    layers = Conv(256, 1)(layers)
    route_1 = layers
    layers = Conv(512, 3, downsample=True)(layers)
    route = layers
    route = Conv(256, 1)(route)
    layers = Conv(256, 1)(layers)
    layers = Residual((256, 256), (1, 3))(layers)
    layers = Residual((256, 256), (1, 3))(layers)
    layers = Residual((256, 256), (1, 3))(layers)
    layers = Residual((256, 256), (1, 3))(layers)
    layers = Residual((256, 256), (1, 3))(layers)
    layers = Residual((256, 256), (1, 3))(layers)
    layers = Residual((256, 256), (1, 3))(layers)
    layers = Residual((256, 256), (1, 3))(layers)
    layers = Conv(256, 1)(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route])

    # block 5

    layers = Conv(512, 1)(layers)
    route_2 = layers
    layers = Conv(1024, 3, downsample=True)(layers)
    route = layers
    route = Conv(512, 1)(route)
    layers = Conv(512, 1)(layers)
    layers = Residual((512, 512), (1, 3))(layers)
    layers = Residual((512, 512), (1, 3))(layers)
    layers = Residual((512, 512), (1, 3))(layers)
    layers = Residual((512, 512), (1, 3))(layers)
    layers = Conv(512, 1)(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route])

    # block 6

    layers = Conv(1024, 1)(layers)
    layers = Conv(512, 1, activation="leaky")(layers)
    layers = Conv(1024, 3, activation="leaky")(layers)
    layers = Conv(512, 1, activation="leaky")(layers)

    # SSP

    layers = SSP()(layers)

    # block 7

    layers = Conv(512, 1, activation="leaky")(layers)
    layers = Conv(1024, 3, activation="leaky")(layers)
    layers = Conv(512, 1, activation="leaky")(layers)

    return (route_1, route_2, layers)


def cspdarknet53_tiny(input_layer):
    # block 1

    layers = Conv(32, 3, activation='leaky', downsample=True)(input_layer)
    layers = Conv(64, 3, activation='leaky', downsample=True)(layers)
    layers = Conv(64, 3, activation='leaky')(layers)
    route = layers
    layers = route_group(layers, 2, 1)
    layers = Conv(32, 3, activation='leaky')(layers)
    route_1 = layers
    layers = Conv(32, 3, activation='leaky')(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route_1])
    layers = Conv(64, 1, activation='leaky')(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([route, layers])
    layers = tf.keras.layers.MaxPool2D(2, 2, 'same')(layers)

    # block 2

    layers = Conv(128 ,3, activation='leaky')(layers)
    route = layers
    layers = route_group(layers, 2, 1)
    layers = Conv(64, 3, activation='leaky')(layers)
    route_1 = layers
    layers = Conv(64, 3, activation='leaky')(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route_1])
    layers = Conv(128, 1, activation='leaky')(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([route, layers])
    layers = tf.keras.layers.MaxPool2D(2, 2, 'same')(layers)

    # block 3

    layers = Conv(256 ,3, activation='leaky')(layers)
    route = layers
    layers = route_group(layers, 2, 1)
    layers = Conv(128, 3, activation='leaky')(layers)
    route_1 = layers
    layers = Conv(128, 3, activation='leaky')(layers)
    layers = tf.keras.layers.Concatenate(axis=-1)([layers, route_1])
    layers = Conv(256, 1, activation='leaky')(layers)
    route_1 = layers
    layers = tf.keras.layers.Concatenate(axis=-1)([route, layers])
    layers = tf.keras.layers.MaxPool2D(2, 2, 'same')(layers)

    layers = Conv(512, 3, activation='leaky')(layers)

    return (route_1, layers)