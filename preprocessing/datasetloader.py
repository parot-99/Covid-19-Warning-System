import numpy as np
import cv2
import os


class DatasetLoader:
    """ Load data into memory from a directory

    Parameters
    ----------
    preprocessors : list, optional
        list of preprocessors to be applied to each image.
        The preprocessor used must return a numpy array.

    """

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, images_path, verbose=1):
        """
        Parameters
        ----------
        images_path : string
            string containing the path to the images directory.
        verbose : int, optional
            how much information to show while loading data.

        Returns
        -------
        tuple
            Returns a tuple containing two numpy arrays
            representing the data values and labels.

        """
        classes = os.listdir(images_path)

        # initialize the list of features and lables
        data = []
        labels = []

        for class_ in classes:
            # images_list = os.listdir(images_path + '/' + class_)
            images_list = os.listdir(os.path.join(images_path, class_))
            class_path = os.path.join(images_path, class_)

            for (i, image_path) in enumerate(images_list):
                #load image and extract the class label
                image = cv2.imread(os.path.join(class_path, image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = class_

                if self.preprocessors is not None:
                    #apply preprocessor to each image
                    for preprocessor in self.preprocessors:
                            image = preprocessor.preprocess(image)

                    data.append(image)
                    labels.append(label)

                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print(f'[INFO] processed {i + 1}/{len(images_list)}')

        return (np.array(data), np.array(labels))
