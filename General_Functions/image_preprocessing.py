def imagePreProcessing(img, brain, label):
    """
    Function for the preprocessing of the images.
  
    It performs multiple tasks: conversion of the image in float32, crop image dimentions to (256x256)
    and normalizes the image intensity by Gaussian normalization over the brain volume.
  
    Parameters
    ----------    
        img (Array): input image.
        brain (Array): brain mask.
        label (Array): label image.
  
    Returns
    ----------
        img (Array of float32): cropped and normalized image.
        label (Array of float32): cropped label image.
    """
    import numpy as np

    # Convert images in float32
    img = np.float32(img)
    brain = np.float32(brain)
    label = np.float32(label)

    # Crop images to standard dimentions (256x256)
    (image_rows, image_columns) = img.shape
    standard_dimentions = 256
    img = img[(int(image_rows/2) - int(standard_dimentions/2)) : (int(image_rows/2) + int(standard_dimentions/2)),
              (int(image_columns/2) - int(standard_dimentions/2)) : (int(image_columns/2) + int(standard_dimentions/2))]
    brain = brain[(int(image_rows/2) - int(standard_dimentions/2)) : (int(image_rows/2) + int(standard_dimentions/2)),
                  (int(image_columns/2) - int(standard_dimentions/2)) : (int(image_columns/2) + int(standard_dimentions/2))]
    label = label[(int(image_rows/2) - int(standard_dimentions/2)) : (int(image_rows/2) + int(standard_dimentions/2)),
                  (int(image_columns/2) - int(standard_dimentions/2)) : (int(image_columns/2) + int(standard_dimentions/2))]
    
    # Gaussian normalization over brain volume
    img -= np.mean(img[brain == 1.0])
    img /= np.std(img[brain == 1.0])
    
    return img, label