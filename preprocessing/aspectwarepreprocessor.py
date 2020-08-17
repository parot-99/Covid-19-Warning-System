import cv2

class AspectWarePreprocessor:
    """Resize data preserving the aspect ratio

    Parameters
    ----------
    width : int
        Target width.
    height : int
        Target height.
    inter : cv2.INTER, optional
        interpolation method used when resizing

    """
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        """
        Parameters
        ----------
        image : numpy array


        Returns
        -------
        numpy array
            Image resized to target shape.

        """
        height, width = image.shape[:2]
        delta_height = 0
        delta_width = 0

        if width < height:
            image = self.__resize_dim(
                image,
                width=self.width,
                inter=self.inter
            )
            delta_height = int((image.shape[0] - self.height) / 2.0)

        else:
            image = self.__resize_dim(
                image,
                height=self.height,
                inter=self.inter
            )
            delta_width = int((image.shape[1] - self.width) / 2.0)

        height, width = image.shape[:2]

        image = image[
            delta_height:height-delta_height,
            delta_width:width-delta_width
        ]

        return cv2.resize(
            image,
            (self.width, self.height),
            interpolation=self.inter
        )

    @staticmethod
    def __resize_dim(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        img_height, img_width = image.shape[:2]

        # If both the width and height are None return the original image
        if width is None and height is None:
            return image

        if width is None:
            # Calculate the ratio of the height and construct the dimensions
            ratio = height / float(img_height)
            dim = (int(img_width * ratio), height)

        else:
            # Calculate the ratio of the width and construct the dimensions
            ratio = width / float(img_width)
            dim = (width, int(img_height * ratio))

        resized = cv2.resize(image, dim, interpolation=inter)

        return resized
