#!/usr/bin/env python3
'''
This script contains useful functions for the training of the network. 

Author: Mattia Ricchi
Date: May 2023
'''
import numpy as np
from os.path import join
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from General_Functions.Nii_Functions import concatenateImages, load_images
from General_Functions.image_preprocessing import imagePreProcessing

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
    np.random.seed(4)
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


def get_test_patients(test_patients_file):
    """
    Read test patient IDs from a file and return them as a NumPy array.

    Parameters
    ----------
    test_patients_file: str
        The name of the file containing test patient IDs.

    Returns
    -------
    numpy.ndarray
        An array of test patient IDs.

    Raises
    ------
    IOError
        If the file specified by `test_patients_file` cannot be opened or read.
    
    Notes
    -----
        The `test_patients_file` must be in the same folder as the script you are executing.
        If the file is in a different folder, please provide as input the whole path to the file.

    Example
    -------
    >>> test_patients = get_test_patients("test_patients.txt")
    >>> print(test_patients)
    [1, 2, 3, 4, 5]
    """
    
    with open(test_patients_file, "r") as file:
        # Read the content of txt file
        content = file.read()
        # Split the content of txt file by spaces
        test_patients_str = content.split()
        # Convert each element to an integer and append it to the list
        test_patients = [int(num) for num in test_patients_str]
    return test_patients
    
    
def has_brain(brain_mask):
    """
    Check if a brain mask image has any non-zero values, indicating the presence of brain.

    Parameters
    ----------
    brain_mask : numpy.ndarray
        A 2D or 3D array representing the brain mask.

    Returns
    -------
    bool
        True if the brain mask has non-zero values, indicating the presence of brain. False otherwise.
    """
    return np.max(brain_mask) > 0.0


def add_to_test_data(TEST_IMAGES, Image_IDs, FLAIR_and_T1W_image, id_):
    """
    Add FLAIR_and_T1W_image and corresponding image ID to the test data.

    Parameters
    ----------
    TEST_IMAGES: numpy.ndarray
        Array representing the test images data, with shape (num_samples, height, width, channels).
    Image_IDs: numpy.ndarray
        Array representing the image IDs for the test images.
    FLAIR_and_T1W_image: numpy.ndarray
        Array representing the FLAIR and T1W image data for a single sample, with shape (height, width, channels).
    id_: str
        The image ID for the FLAIR_and_T1W_image.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Updated TEST_IMAGES array and Image_IDs array after adding the new sample.
    """
    
    TEST_IMAGES = np.append(TEST_IMAGES, FLAIR_and_T1W_image[np.newaxis, ...], axis=0)
    Image_IDs = np.append(Image_IDs, id_)
    
    return TEST_IMAGES, Image_IDs


def add_to_train_data(TRAIN_IMAGES, TRAIN_LABELS, FLAIR_and_T1W_image, label_image):
    """
    Add data to the training arrays.

    Parameters
    ----------
    TRAIN_IMAGES: ndarray
        Array containing the training images.
    TRAIN_LABELS: ndarray
        Array containing the training labels.
    FLAIR_and_T1W_image: ndarray
        FLAIR and T1W image to be added to the training data.
    label_image: 2darray
        Label map to be added to the labels data.

    Returns
    -------
    TRAIN_IMAGES: ndarray
        Updated array containing the training images.
    TRAIN_LABELS: ndarray
        Updated array containing the training labels.
    """
    
    TRAIN_IMAGES = np.append(TRAIN_IMAGES, FLAIR_and_T1W_image[np.newaxis, ...], axis = 0)
    TRAIN_LABELS = np.append(TRAIN_LABELS, label_image[np.newaxis, ...], axis = 0)
            
    return TRAIN_IMAGES, TRAIN_LABELS


def build_train_test_data(data_path, test_patients, labeled_ids, id_, TEST_IMAGES, TRAIN_IMAGES, TRAIN_LABELS, Image_IDs):
    """
    This function classifies the input images as part of the training or testing datasets based on the image ID.
    Input slices with no brain anre skipped by the function.
    If the input slice is classified as a training image and it contains white matter lesion, data augemntation is applied 9 times.

    Parameters
    ----------
    data_path: str
        Path to the dataset folder
    test_patients: np.array
        Array containing the id of patients that will be used for testing
    labeled_ids: np.array
        Array containing the id of slices with labeled lesions
    id_: str 
        The ID of the image that needs to be classified
    TEST_IMAGES: np.ndarray
        Array containing test images to which the new image will be appended
    TRAIN_IMAGES: np.ndarray
        Array containing training images to which the new image will be appended
    TRAIN_LABELS: np.ndarray
        Array containing the labels of training images to which the new label will be appended
    Image_IDs: np.array
        Array containing the IDs of all the images that will be used for testing

    Returns
    -------
    TEST_IMAGES: np.ndarray
        Updated array containing test images
    TRAIN_IMAGES: np.ndarray
        Updated array containing training images
    TRAIN_LABELS: np.ndarray
        Updated array containing the labels of training images
    Image_IDs: np.array
        Updated array containing the IDs of all the images that will be used for testing
    """
    
    # Define necessary paths
    flair_path = join(data_path, 'OnlyBrain/flair/')
    t1w_path = join(data_path, 'OnlyBrain/t1w/')
    label_path = join(data_path, 'OnlyBrain/label/')
    brain_path = join(data_path, 'brain/')
    
    # Load the images and labels
    flair_image, t1w_image, label_image, brain_mask = load_images(flair_path, t1w_path, label_path, brain_path, id_)

    # Skip the images in which there is no brain
    if not has_brain(brain_mask):
        return TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, Image_IDs

    # Preprocess the images and labels
    (flair_image, label_image) = imagePreProcessing(flair_image, brain_mask, label_image)
    (t1w_image, label_image) = imagePreProcessing(t1w_image, brain_mask, label_image)
    
    # Concatenate FLAIR and T1W images
    FLAIR_and_T1W_image = concatenateImages(flair_image, t1w_image)

    # Sort images based on the patient number
    patient_number = int(id_[7:10])
    
    if patient_number in test_patients:
        # Image will be used for testing
        TEST_IMAGES, Image_IDs = add_to_test_data(TEST_IMAGES, Image_IDs, FLAIR_and_T1W_image, id_)
        
    else:    
        # Image is classified as a training image
        TRAIN_IMAGES, TRAIN_LABELS = add_to_train_data(TRAIN_IMAGES, TRAIN_LABELS, FLAIR_and_T1W_image, label_image)
    
        # Apply data augmentation 10 times if there are labeled lesions in the slice
        if id_[:14] in labeled_ids:
            labelImg = label_image[0, :, :, 0]
            for _ in range(9):
                flairAug, t1Aug, labelAug = dataAugmentation(flair_image, t1w_image, labelImg)
                FLAIR_and_T1W_image = concatenateImages(flairAug, t1Aug)
                FLAIR_and_T1W_image = FLAIR_and_T1W_image[np.newaxis, ...]
                labelAug = labelAug[np.newaxis, ..., np.newaxis]
                TRAIN_IMAGES = np.append(TRAIN_IMAGES, FLAIR_and_T1W_image, axis = 0)
                TRAIN_LABELS = np.append(TRAIN_LABELS, labelAug, axis = 0)
                
    return TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, Image_IDs