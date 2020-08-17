import argparse
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


def test_image(args):
    model = load_model('resnet.hdf5')

    base_image = cv2.imread(args['image'])

    image = base_image
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) 
    image = preprocess_input(image)

    prediction = model.predict(image)
    prediction = prediction[0][0]

    label = 'no-mask' if prediction >= 0.5 else 'mask'

    prediction = prediction if prediction >= 0.5 else 1 - prediction

    color = (0, 0, 255) if label == 'no-mask' else (0, 255, 0)

    cv2.putText(
        base_image,
        f'{label}: {100 * prediction:.2f}%',
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )
    cv2.imshow('Image', base_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser() 
    arg_parser.add_argument('-i', '--image')
    args = vars(arg_parser.parse_args())

    test_image(args)
