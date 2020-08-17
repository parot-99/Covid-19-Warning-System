from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    """Used to transform images to numpy arrays"""
    def __init__(self, data_format=None):
        self.data_format = data_format

    def preprocess(self, image):
        return img_to_array(image, self.data_format)