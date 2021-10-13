import numpy as np
from .iou import calculate_iou


def get_image_results(predicted_objs, ground_truths, iou_thresh):
    predicted_objs_indices = range(len(predicted_objs))
    ground_truths_indices = range(len(ground_truths))

    if len(predicted_objs_indices) == 0:
        tp = 0
        fp = 0
        fn = len(ground_truths)
        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
    if len(ground_truths_indices) == 0:
        tp = 0
        fp = len(predicted_objs)
        fn = 0
        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }

    predicted_idx = []
    truths_idx = []
    ious = []

    for prediction_id, predicted_obj in enumerate(predicted_objs):
        for truth_id, ground_truth in enumerate(ground_truths):

            iou = calculate_iou(predicted_obj, ground_truth)

            if iou > iou_thresh:
                predicted_idx.append(prediction_id)
                truths_idx.append(truth_id)
                ious.append(iou)

    sorted_ious = np.argsort(ious)[::1]

    if len(sorted_ious) == 0:
        tp = 0
        fp = len(predicted_objs)
        fn = len(ground_truths)

        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }

    else:
        prediction_match_idx = []
        truths_match_idx = []

        for id in sorted_ious:
            prediction_id = predicted_idx[id]
            truth_id = truths_idx[id]

            if (prediction_id not in prediction_match_idx) and (truth_id not in truths_match_idx):
                prediction_match_idx.append(prediction_id)
                truths_match_idx.append(truth_id)

        tp = len(truths_match_idx)
        fp = len(predicted_objs) - len(prediction_match_idx)
        fn = len(ground_truths) - len(truths_match_idx)

    results = {"true_positive": tp, "false_positive": fp, "false_negative": fn}

    return results
