"""Augment and load images using keras ImageDataGenerator"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataAugmenter:
    """Load and preprocess data using Keras
    ImageDataGenerator and the flow from directory method

    Parameters
    ----------
    directory : string
        location of dataset.
    target_size : tuple
        tuple containing the new width and height of target size.
    batch_size : int
        number of batches to process at each epoch.
    class_mode : string
        binary or categorical.
    preprocessor : function
        preprocessing function from keras
    rescale : bool
        if rescale is true, rescale images between 0 and 1, else
        use preprocessing function.
    """
    def __init__(self,
                 directory,
                 target_size,
                 batch_size,
                 class_mode,
                 preprocessor=None, 
                 rescale=True):
        self.directory = directory
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.preprocessor = preprocessor
        self.rescale = 1. / 255 if rescale == True else None

    def flow_from_directory(self):
        """Return a data and test generator used in fit
        methode to flow images from specified directory
        """

        data_generator = ImageDataGenerator(
            rescale=self.rescale,
            preprocessing_function=self.preprocessor,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        test_generator = ImageDataGenerator(
            preprocessing_function=self.preprocessor, 
            rescale=self.rescale
        )

        train_generator = data_generator.flow_from_directory(
            self.directory + '/train',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode
        )

        validation_generator = test_generator.flow_from_directory(
            self.directory + '/validate',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
        )

        return train_generator, validation_generator
