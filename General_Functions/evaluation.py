#!/usr/bin/env python3
'''
This script contains a useful function for the evaluation of the performance fo the network. 

Author: Mattia Ricchi
Date: May 2023
'''

def get_evaluation_metrics(true_image, predicted_image):
    """
    The function computes the dice coefficient, precision, recall, and F1-score between the true image and the predicted image.

    Parameters
    ----------
    true_image: numpy.ndarray
        A 2D array containing the ground truth image.
    predicted_image: numpy.ndarray
        A 2D array containing the image predicted by the network.

    Returns
    -------
    tuple 
        Tuple containing the evaluation metrics (dice coefficient, precision, recall, F1-score).

    Raises
    ------
    AssertionError: If the true_image and predicted_image have different shapes.

    Notes
    -----
    It uses the dice_coef_for_training function from the Training_Functions module for calculating the dice coefficient.
    The precision, recall, and F1-score are calculated using the micro-average method with support for zero division.

    Example
    -------
    >>> import numpy as np
    >>> from Training_Functions import dice_coef_for_training
    >>> from sklearn.metrics import precision_score, recall_score, f1_score
    >>> true_image = np.array([1, 0, 1, 1])
    >>> predicted_image = np.array([1, 1, 0, 1])
    >>> dsc, precision, recall, f1 = get_evaluation_metrics(true_image, predicted_image)
    >>> print(dsc, precision, recall, f1)
    0.6666666666666666 0.6666666666666666 0.6666666666666666 0.6666666666666666
    """
    assert true_image.shape == predicted_image.shape, "True image and predicted image must have the same shape."

    from Training_Functions import dice_coef_for_training
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    dsc = dice_coef_for_training(true_image, predicted_image)
    precision = precision_score(true_image, predicted_image, average='micro', zero_division=True)
    recall = recall_score(true_image, predicted_image, average='micro', zero_division=True)
    f1 = f1_score(true_image, predicted_image, average='micro', zero_division=True)
    
    return dsc, precision, recall, f1
