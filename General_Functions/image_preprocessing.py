'''
This script contains all the necessary functions for the preprocessing of the MR images. 

Author: Mattia Ricchi
Date: May 2023
'''
def crop_image(image, standard_dimentions = 256):
    '''
    Function to crop the input image to standard dimentions (256x256)
    '''
    (image_rows, image_columns) = image.shape

    cropped_image = image[(int(image_rows/2) - int(standard_dimentions/2)) : (int(image_rows/2) + int(standard_dimentions/2)),
              (int(image_columns/2) - int(standard_dimentions/2)) : (int(image_columns/2) + int(standard_dimentions/2))]

    return cropped_image


def gaussian_normalisation(image, brain_mask):
    '''
    Function to perform gaussian normalisation of pixel intensity
    '''
    import numpy as np
    
    # Compute the mean and standard deviation of the image within the brain mask
    image_mean = np.mean(image[brain_mask == 1.0])
    image_std = np.std(image[brain_mask == 1.0])

    # Normalize the image using a Gaussian distribution with zero mean and unit variance
    normalised_image = (image - image_mean) / image_std
    
    return normalised_image


def float32_converter(image):
    '''
    Function to convert the input image to numpy float32 array
    '''
    import numpy as np
    image_float32 = np.float32(image)

    return image_float32


def imagePreProcessing(image, brain_mask, label):
    """
    Function for the preprocessing of the images:
    This function converts the input images into float32, crops images dimentions to (256x256)
    and normalizes the brain image intensity by Gaussian normalization over the brain volume.

    Parameters
    ----------    
        image (Array): input brain image.
        brain_mask (Array): brain mask.
        label (Array): label image.
  
    Returns
    ----------
        image (Array of float32): cropped and normalized brain image.
        label (Array of float32): cropped label image.
    """

    # Convert images in float32
    image = float32_converter(image)
    brain_mask = float32_converter(brain_mask)
    label = float32_converter(label)

    # Crop images to standard dimentions (256x256)
    image = crop_image(image)
    brain_mask = crop_image(brain_mask)
    label = crop_image(label)

    # Gaussian normalization over brain volume
    image = gaussian_normalisation(image, brain_mask)
    
    return image, label