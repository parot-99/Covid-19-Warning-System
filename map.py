from tensorflow._api.v2 import image
from metrics.results import get_image_results
from yolo.yolo_mask import YoloMask
import numpy as np
import os

def main():
    yolo = YoloMask()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    images_path = './../../darknet/data/mask-dataset/images/test/'

    images = os.listdir(images_path)

    for image in images:
        image_name, ext = image.split('.')
        if ext == 'txt':
            continue

        image = images_path + image
        txt = images_path + image_name + '.txt'

        results = predict_and_get_results(yolo, image, txt)
        total_tp += results['true_positive']
        total_fp += results['false_positive']
        total_fn += results['false_negative']

    print(total_tp)
    print(total_fp)
    print(total_fn)


def predict_and_get_results(yolo, image, txt):
    bounding_boxes = yolo.get_bounding_boxes(image)
    ground_truths = []

    with open(txt, "r") as file:
        lines = file.readlines()

        for line in lines:
            line = line.split(' ')
            line = line[1:]
            line[3] = line[3].split('\n')[0]
            center_x = float(line[0])
            center_y = float(line[1])
            width = float(line[2])
            height = float(line[3])
            ground_truth = [
                round(center_x - (width / 2), 6),
                center_y - (height / 2),
                center_x + (width / 2),
                center_y + (height / 2)
            ]
            ground_truths.append(ground_truth)


    boxes, scores, classes, box_count = bounding_boxes

    predicted_objs = boxes[0][0:box_count[0]]
    ground_truths = np.asarray(ground_truths, dtype=np.float32)

    return get_image_results(predicted_objs, ground_truths, 0.5)


if __name__ == '__main__':
    # predict_and_get_results()
    main()