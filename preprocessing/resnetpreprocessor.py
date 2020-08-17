from tensorflow.keras.applications.resnet50 import preprocess_input


class ResNetPreprocessor:
    @staticmethod
    def preprocess(image):
        return  preprocess_input(image)
