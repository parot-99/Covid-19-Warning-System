from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential


class MiniVGGNet:
    @staticmethod
    def build(input_shape, classes):
        """Build a neural network using VGG-like architecture"""
        activation = 'sigmoid' if classes == 1 else 'softmax'

        model = Sequential()
        model.add(
            Conv2D(32, (3,3),
            padding='same',
            input_shape=input_shape,
            activation='relu')
        )
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation=activation))

        return model
