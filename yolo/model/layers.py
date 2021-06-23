import tensorflow as tf


class Identity(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)

        training = tf.logical_and(training, self.trainable)

        return super().call(x, training)


class Conv(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel,
        downsample=False,
        bn=True,
        activation="mish",
    ):
        super().__init__()
        self.bn = bn
        self.downsample = downsample
        self.padding = "valid" if downsample else "same"
        self.strides = 2 if downsample else 1

        if self.downsample:
            self.zero_padding = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=self.strides,
            padding=self.padding,
            use_bias=not bn,
            # kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            kernel_regularizer=tf.keras.regularizers.L2(l2=0.005),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )

        if self.bn:
            self.batch_normalization = tf.keras.layers.BatchNormalization(
                epsilon=1e-5, momentum=0.9
            )

        self.activation = tf.keras.layers.Activation("mish")

        if activation == "leaky":
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.1)

        if activation == "linear":
            self.activation = tf.keras.layers.Activation("linear")

    def call(self, input_layer, training=False):
        output = input_layer

        if self.downsample:
            output = self.zero_padding(output)

        output = self.conv(output)

        if self.bn:
            output = self.batch_normalization(output, training=training)

        output = self.activation(output)

        return output


class Residual(tf.keras.layers.Layer):
    def __init__(self, filters, kernels, activation="mish"):
        super().__init__()
        self.conv1 = Conv(filters[0], kernels[0], activation=activation)
        self.conv2 = Conv(filters[1], kernels[1], activation=activation)
        self.activate = tf.keras.layers.Activation("linear")

    def call(self, input_layer):
        output = self.conv1(input_layer)
        output = self.conv2(output)
        residual_output = tf.keras.layers.add([input_layer, output])
        residual_output = self.activate(residual_output)

        return residual_output


class MaxPool(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides):
        super().__init__()
        self.pool = tf.keras.layers.MaxPool2D(
            pool_size=pool_size, strides=strides, padding="same"
        )

    def call(self, input_layer):
        output = self.pool(input_layer)

        return output


class SSP(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.pool_1 = MaxPool(5, 1)
        self.pool_2 = MaxPool(9, 1)
        self.pool_3 = MaxPool(13, 1)

    def call(self, input_layer):
        output_1 = self.pool_1(input_layer)
        output_2 = self.pool_2(input_layer)
        output_3 = self.pool_3(input_layer)

        output = tf.keras.layers.Concatenate(axis=-1)(
            [output_3, output_2, output_1, input_layer]
        )

        return output


class Upsample(tf.keras.layers.Layer):
    def __init__(self, stride):
        super().__init__()
        self.upsample = tf.keras.layers.UpSampling2D(
            size=stride,
            interpolation="bilinear",
        )

    def call(self, input_layer):
        output = self.upsample(input_layer)

        return output


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


class Mish(tf.keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super().__init__(activation, **kwargs)
        self.__name__ = "mish"


def mish(x):
    return x * tf.keras.backend.tanh(tf.keras.backend.softplus(x))

tf.keras.utils.get_custom_objects().update({"mish": Mish(mish)})


class Route(tf.keras.layers.Layer):
    def __init__(self, route):
        super().__init__()
        self.route = route

    def call(self, layers):
        layer = tf.concat([layers, self.route], axis=-1)

        return layer
