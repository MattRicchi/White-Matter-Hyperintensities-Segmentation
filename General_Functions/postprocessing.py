#!/usr/bin/env python3
'''
This script contains a useful function for the evaluation of the performance fo the network. 

Author: Mattia Ricchi
Date: May 2023
'''
import numpy as np
import cv2
from General_Functions.image_preprocessing import crop_image
from General_Functions.Training_Functions import dice_coef_for_training
from sklearn.metrics import precision_score, recall_score, f1_score

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
    
    dsc = dice_coef_for_training(true_image, predicted_image)
    precision = precision_score(true_image, predicted_image, average='micro', zero_division=True)
    recall = recall_score(true_image, predicted_image, average='micro', zero_division=True)
    f1 = f1_score(true_image, predicted_image, average='micro', zero_division=True)
    
    return dsc, precision, recall, f1


def imagePostProcessing(img, label):
    """
    Function for the postprocessing of the images:
    The function converts the input images to float32 data type;
    It then crops the images to the standard dimensions of 256x256;
    The image intensity is normalized using the cv2.normalize function with alpha=0 and beta=255.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    label : numpy.ndarray
        Label image.

    Returns
    -------
    numpy.ndarray
        Cropped and normalized image.
    numpy.ndarray
        Cropped label image.

    Notes
    -----
    The function converts the input images to float32 data type.
    It then crops the images to the standard dimensions of 256x256.
    The image intensity is normalized using the cv2.normalize function with alpha=0 and beta=255.
    The resulting image and label are returned as a tuple.

    Example
    -------
    >>> import numpy as np
    >>> from image_processing import imagePreProcessing

    >>> img = np.random.rand(512, 512)  # Sample input image
    >>> label = np.random.randint(0, 2, (512, 512))  # Sample label image

    >>> processed_img, processed_label = imagePreProcessing(img, label)
    >>> print(processed_img.shape)
    (256, 256)
    >>> print(processed_label.shape)
    (256, 256)
    """

    # Convert images to float32
    img = np.float32(img)
    label = np.float32(label)

    # Crop to standard dimensions (256x256)
    img = crop_image(img)
    label = crop_image(label)

    # Normalize image intensity 0 to 255
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img, label
