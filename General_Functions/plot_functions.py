#!/usr/bin/env python3
'''
This script contains the functions to display the images

Author: Mattia Ricchi
Date: June 2023
'''

import os
from os.path import join
import matplotlib.pyplot as plt 
from General_Functions.Nii_Functions import readImage
from General_Functions.postprocessing import imagePostProcessing

def read_images_to_plot(patient_number, slice_number):
    """
    This function reads and retrieves the images for a specific patient and slice in order to plot them. 
    The images to be read include the FLAIR image, the ground truth image, and the result image.
    It assumes a specific directory structure for the data and results. 

    Parameters
    ----------
    patient_number: int
        The number of the patient for which the images are being read.
    slice_number: int
        The number of the slice within the patient's data.

    Returns
    -------
    flair_image: ndarray
        The image data for the FLAIR image.
    label: ndarray
        The image data for the ground truth label.
    result: ndarray
        The image data for the result.

    Notes
    -----
    - The function assumes that the FLAIR images are stored in the 'flair/' subdirectory of the 'DATABASE' directory, 
        and the ground truth label images are stored in the 'label/' subdirectory of the 'DATABASE' directory.
    - The image data is processed using the `imagePostProcessing` function.

    Example
    --------
    >>> flair_image, label, result = read_images_to_plot(10, 5)
    """
    # Set up necessary paths for data and results
    data_path = join(os.getcwd(), 'DATABASE')
    results_path = join(os.getcwd(), 'Results')
    flair_path = join(data_path, 'flair/')
    label_path = join(data_path, 'label/')
    
    # Reconstruct the name of the image you want to display
    SLICE_DECIMATE_IDENTIFIER = 3
    image_name = f'volume-{str(patient_number).zfill(SLICE_DECIMATE_IDENTIFIER)}-{str(slice_number).zfill(SLICE_DECIMATE_IDENTIFIER)}.nii'
    
    # Read flair, label and result images
    flair_image = imagePostProcessing(readImage(join(flair_path, image_name)))
    label = imagePostProcessing(readImage(join(label_path, image_name)))
    result = imagePostProcessing(readImage(join(results_path, image_name)))
    
    return flair_image, label, result

def plot_images(patient_number, slice_number):
    """
    Plot FLAIR image, ground truth label, and segmentation result for a specific patient and slice.
    This function reads the necessary images using the `read_images_to_plot` function and plots them in a single figure. 
    The figure contains three subplots: the FLAIR image, the ground truth label, and the segmentation result.

    Parameters
    ----------
    patient_number: int
        The number of the patient for which the images are being plotted.
    slice_number: int
        The number of the slice within the patient's data.

    Returns
    -------
    None

    Example
    --------
    >>> plot_images(10, 5)
    """
    # Read images to plot
    flair_image, label, result = read_images_to_plot(patient_number, slice_number)
    
    # Define figure size and title
    fig = plt.figure(figsize=(15, 6))
    plt.title(f'Volume {patient_number} - Slice {slice_number}', fontsize = 25)
    plt.axis('off')

    rows = 1
    columns = 3

    # Add flair image at the 1st position
    fig.add_subplot(rows, columns, 1)
    plt.imshow(flair_image.T, cmap = 'gray')
    plt.title("Flair Image", fontsize = 16)

    # Add ground truth image at the 2nd position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(label.T, cmap = 'gray')
    plt.title("Label Image", fontsize = 18)

    # Add segmentation result at the 3rd position
    fig.add_subplot(rows, columns, 3)
    plt.imshow(result.T, cmap = 'gray')
    plt.title("Segmentation Result", fontsize = 18)
    
    return