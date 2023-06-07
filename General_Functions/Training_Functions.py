#!/usr/bin/env python3
'''
This script contains useful functions for the training of the network. 

Author: Mattia Ricchi
Date: May 2023
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

def dataAugmentation(flair, t1, label):
    """
    This function applies random transformations to the input images to increase the size of the dataset
    and improve the performance of the model. The transformations include rotation, shear, and zoom.

    Parameters
    ----------
    flair: numpy.ndarray
        A 2D array containing the FLAIR image.
    t1: numpy.ndarray
        A 2D array containing the T1 image.
    label: numpy.ndarray
        A 2D array containing the ground truth segmentation label.

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Three 2D arrays containing the augmented FLAIR, T1, and label images.

    Example
    -------
    >>> flair = np.array([[1, 2], [3, 4]])
    >>> t1 = np.array([[5, 6], [7, 8]])
    >>> label = np.array([[0, 1], [1, 0]])
    >>> flair_aug, t1_aug, label_aug = dataAugmentation(flair, t1, label)
    >>> flair_aug
    array([[1., 1.],
           [1., 1.]])
    >>> t1_aug
    array([[7., 7.],
           [7., 7.]])
    >>> label_aug
    array([[0., 0.],
           [0., 0.]])
    """

    im_gen = ImageDataGenerator()

    # Randomly generate the transformation parameters
    theta = np.random.uniform(-15, 15)
    shear = np.random.uniform(-.1, .1)
    zx, zy = np.random.uniform(.9, 1.1, 2)

    # Apply the transformations to the images
    flairAug = im_gen.apply_transform(x=flair[..., np.newaxis], transform_parameters={'theta': theta, 'shear': shear, 'zx': zx, 'zy': zy})
    t1Aug = im_gen.apply_transform(x=t1[..., np.newaxis], transform_parameters={'theta': theta, 'shear': shear, 'zx': zx, 'zy': zy})
    labelAug = im_gen.apply_transform(x=label[..., np.newaxis], transform_parameters={'theta': theta, 'shear': shear, 'zx': zx, 'zy': zy})

    return flairAug[:, :, 0], t1Aug[:, :, 0], labelAug[:, :, 0]


def dice_coef_for_training(y_true, y_pred):
    """
    Computes the dice coefficient between the ground truth labels and the predicted labels for use in training.

    Parameters
    ----------
    y_true: array-like
        The ground truth labels.
    y_pred: array-like
        The predicted labels.

    Returns
    -------
    dice_coef: float
        The dice coefficient between the ground truth labels and the predicted labels.

    Notes
    -----
    The dice coefficient is a common evaluation metric for segmentation tasks. It measures the overlap between the predicted
    labels and the ground truth labels, and ranges from 0 to 1, with 1 indicating a perfect match. The dice coefficient is 
    computed as:

        dice_coef = (2 * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)

    where `intersection` is the element-wise product between the ground truth labels and the predicted labels, `smooth` is 
    a smoothing factor to avoid division by zero, `sum(y_true_f)` and `sum(y_pred_f)` are the sum of the ground truth 
    labels and the predicted labels, respectively.

    Example
    -------
    >>> import numpy as np
    >>> y_true = np.array([[0, 1], [1, 1]])
    >>> y_pred = np.array([[0, 0], [1, 1]])
    >>> dice_coef_for_training(y_true, y_pred)
    0.8
    """

    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice_coef


def dice_coef_loss(y_true, y_pred):
    """
    Calculates the Dice coefficient loss between the ground truth segmentation maps (y_true) and predicted maps (y_pred).
    
    Parameters
    ----------
    y_true: numpy.ndarray
        The ground truth segmentation maps with shape (batch_size, height, width).
    y_pred: numpy.ndarray
        The predicted segmentation maps with shape (batch_size, height, width).

    Returns
    -------
    float
        The Dice coefficient loss.

    Notes
    -----
    The Dice coefficient is a measure of the overlap between two sets. It ranges from 0 (no overlap) to 1 (perfect overlap).
    The Dice coefficient loss is defined as 1 - the Dice coefficient. The loss is used as the objective function during model training.

    Examples
    --------
    >>> y_true = np.array([[1, 1, 0], [0, 1, 1]])
    >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]])
    >>> dice_coef_loss(y_true, y_pred)
    0.16666666666666669
    """

    return -dice_coef_for_training(y_true, y_pred)


def get_crop_shape(target, refer):
    '''
    Calculate the crop shape necessary to match the dimensions of two tensors.

    Parameters
    -----------
    target: tensor
        The tensor to be cropped.
    refer: tensor
        The tensor whose dimensions will be matched.

    Returns
    --------
    Tuple of two tuples of integers, each containing two values:
    - The first tuple contains the crop amounts for the height dimension (second dimension of the tensors).
    - The second tuple contains the crop amounts for the width dimension (third dimension of the tensors).

    Raises
    -------
    AssertionError: if the target tensor's dimensions are smaller than the reference tensor's dimensions.

    Example
    --------
    >>> target = tf.zeros([4, 10, 8, 3])
    >>> refer = tf.zeros([4, 8, 6, 3])
    >>> crop_shape = get_crop_shape(target, refer)
    >>> print(crop_shape)
    ((1, 1), (0, 0))
    '''
    cw = target.get_shape()[2] - refer.get_shape()[2]
    
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = target.get_shape()[1] - refer.get_shape()[1]

    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)


def scheduler(epoch, learning_rate):
    '''
    This function uses the exponential decay formula to update the learning rate:
        new_learning_rate = initial_learning_rate * exp(-0.1 * epoch)
    After epoch 10, the learning rate decreases exponentially with a factor of 0.1 at each epoch.

    Parameters
    -----------
    epoch: int
        Current epoch number
    learning_rate: float
        Current learning rate value

    Returns
    --------
    float
        Updated learning rate value based on the epoch number

    Example
    --------
    # Define learning rate scheduler function
    def scheduler(epoch, learning_rate):
        if epoch < 10:
            return learning_rate
        else:
            return learning_rate * tf.math.exp(-0.1)

    # Set the learning rate scheduler to be used in the training
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    '''

    if epoch < 10:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)
