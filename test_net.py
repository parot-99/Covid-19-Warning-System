import argparse
import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from preprocessing.aspectwarepreprocessor import AspectWarePreprocessor
from preprocessing.resnetpreprocessor import ResNetPreprocessor
from preprocessing.imagetoarray import ImageToArrayPreprocessor
from preprocessing.datasetloader import DatasetLoader


def test_data(resnet):
    """Test a model on a test dataset"""
    aspect_ware_preprocessor = AspectWarePreprocessor(int(args['size']),
                                                      int(args['size']))
    image_preprocessor = ImageToArrayPreprocessor()
    data_loader = DatasetLoader(preprocessors=[
        aspect_ware_preprocessor,
        image_preprocessor
    ])

    if resnet == 'True':
        data_loader = DatasetLoader(preprocessors=[
            aspect_ware_preprocessor,
            image_preprocessor,
            ResNetPreprocessor()
        ])

    data, labels = data_loader.load(args['data'], verbose=100)

    if resnet != 'True':
        data = data.astype('float32') / 255.0


    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)

    model = load_model(args['model'])

    predictions = model.predict(data, batch_size=32)

    for i, prediction in enumerate(predictions):
        if prediction > 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0

    print(classification_report(labels,
                                predictions,
                                target_names=['mask', 'no-mask']))


def test_image(resnet):
    """Test model on 1 image and show the result using open-cv"""
    model = load_model(args['model'])

    base_image = cv2.imread(args['image'])

    image = base_image
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image = cv2.resize(image, (int(args['size']), int(args['size'])))
    image = np.expand_dims(image, axis=0)

    if resnet != 'True':
        image = image.astype('float32') / 255.0

    if resnet == 'True':
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
    arg_parser.add_argument('-d', '--data')
    arg_parser.add_argument('-i', '--image')
    arg_parser.add_argument(
        '-m',
        '--model',
        default='trained_nets/ResNet/net.hdf5'
    )
    arg_parser.add_argument('-r', '--resnet', default='True')
    arg_parser.add_argument('-s', '--size', default=224)
    args = vars(arg_parser.parse_args())

    if args['data'] is not None:
        test_data(args['resnet'])

    if args['image'] is not None:
        test_image(args['resnet'])
