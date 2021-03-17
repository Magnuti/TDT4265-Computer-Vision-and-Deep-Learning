import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    [xmin_pred, ymin_pred, xmax_pred, ymax_pred] = prediction_box
    [xmin_gt, ymin_gt, xmax_gt, ymax_gt] = gt_box

    # Check if there is no overlap
    if(xmax_gt < xmin_pred or xmax_pred < xmin_gt or ymax_gt < ymin_pred or ymax_pred < ymin_gt):
        return 0

    # Compute intersection
    xmin_intersection = max(xmin_pred, xmin_gt)
    ymin_intersection = max(ymin_pred, ymin_gt)
    xmax_intersection = min(xmax_pred, xmax_gt)
    ymax_intersection = min(ymax_pred, ymax_gt)

    intersection = (xmax_intersection - xmin_intersection) * \
        (ymax_intersection - ymin_intersection)

    area_pred = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    area_gt = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)

    # Compute union
    union = area_pred + area_gt - intersection
    iou = intersection / union
    assert iou >= 0 and iou <= 1, iou
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    divisor = num_tp + num_fp
    if divisor == 0:
        return 1

    return num_tp / divisor


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    divisor = num_tp + num_fn
    if divisor == 0:
        return 0

    return num_tp / divisor


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        iou_threshold: float
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    matches = {}
    for i in range(prediction_boxes.shape[0]):
        for k in range(gt_boxes.shape[0]):
            iou = calculate_iou(prediction_boxes[i], gt_boxes[k])
            if iou >= iou_threshold:
                matches[(i, k)] = iou

    found_matches = []
    found_pred = []
    found_gt = []

    # Sort all matches on IoU in descending order
    # Find all matches with the highest IoU threshold
    for key, value in sorted(matches.items(), key=lambda x: x[1], reverse=True):
        # Skip pairs that are already added since they are worse
        if key in found_matches:
            continue

        found_matches.append(key)
        found_pred.append(prediction_boxes[key[0]])
        found_gt.append(gt_boxes[key[1]])

    return np.array(found_pred), np.array(found_gt)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        iou_threshold: float
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    total_predictions = prediction_boxes.shape[0]
    total_ground_boxes = gt_boxes.shape[0]

    prediction_boxes, gt_boxes = get_all_box_matches(
        prediction_boxes, gt_boxes, iou_threshold)

    num_tp = prediction_boxes.shape[0]
    num_fp = total_predictions - num_tp
    num_fn = total_ground_boxes - num_tp

    return {"true_pos": num_tp, "false_pos": num_fp, "false_neg": num_fn}


def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        iou_threshold: float
    Returns:
        tuple: (precision, recall). Both float.
    """
    assert len(all_prediction_boxes) == len(all_gt_boxes)

    num_tp = 0
    num_fp = 0
    num_fn = 0

    for i in range(len(all_prediction_boxes)):
        image_results = calculate_individual_image_result(
            all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)

        num_tp += image_results["true_pos"]
        num_fp += image_results["false_pos"]
        num_fn += image_results["false_neg"]

    return calculate_precision(num_tp, num_fp, num_fn), calculate_recall(num_tp, num_fp, num_fn)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        confidence_scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
        iou_threshold: float
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)

    precisions = []
    recalls = []
    raise NotImplementedError()
    # for threshold in confidence_thresholds:
    #     pass

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    average_precision = 0
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(
        precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
