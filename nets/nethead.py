from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


class NetHead:
    @staticmethod
    def build(base_model, classes, dense_output):
        activation = 'sigmoid' if classes == 1 else 'softmax'

        head_model = base_model.output
        head_model = AveragePooling2D(pool_size=(7,7))(head_model)
        head_model = Flatten(name='flatten')(head_model)
        head_model = Dense(dense_output, activation='relu')(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(classes, activation=activation)(head_model)

        return head_model
