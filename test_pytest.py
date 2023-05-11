'''
This script contains all the test functions used to test the code

Author: Mattia Ricchi
Date: May 2023
'''
#!/usr/bin/env python3

def test_readImage():
    from General_Functions.Nii_Functions import readImage
    import numpy as np
    import nibabel as nib
    import os
    from pathlib import Path

    # First define a test image array and a test file name and path
    test_img = np.random.rand(10, 10, 10)
    fname = "test_image"
    path = os.path.join(os.getcwd(), 'test_folder')

    # Save the test image using nibabel
    test_img_nib = nib.Nifti1Image(test_img, np.eye(4))
    nib.save(test_img_nib, os.path.join(path, f"{fname}.nii"))

    # Load the saved image using the readImage function
    loaded_img = readImage(os.path.join(path, f"{fname}.nii"))

    # Check that the loaded image matches the original test image
    assert np.allclose(test_img, loaded_img)


def test_saveSlice():
    from General_Functions.Nii_Functions import saveSlice
    import numpy as np
    import nibabel as nib
    import os

    # First define a test image array and a test file name and path
    test_img = np.random.rand(10, 10, 10)
    fname = "test_image"
    path = os.path.join(os.getcwd(), 'test_folder')

    # Save the test image using the saveSlice function
    saveSlice(test_img, fname, path)

    # Load the saved image using nibabel
    loaded_img = nib.load(os.path.join(path, f"{fname}.nii")).get_fdata()

    # Check that the loaded image matches the original test image
    assert np.allclose(test_img, loaded_img)

def test_concatenateImages():
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np

    # First define two test images
    test_img_1 = np.random.rand(10, 10)
    test_img_2 = np.random.rand(10, 10)

    # Concatenate the two images using the concatenateImages function
    concatenated_img = concatenateImages(test_img_1, test_img_2)

    # Check that the shape of the concatenated image is correct
    assert concatenated_img.shape == (10, 10, 2)

    # Check that the first channel of the concatenated image matches the first input image
    assert np.array_equal(concatenated_img[..., 0], test_img_1)

    # Check that the second channel of the concatenated image matches the second input image
    assert np.array_equal(concatenated_img[..., 1], test_img_2)

def test_dataAugmentation():
    from General_Functions.Training_Functions import dataAugmentation
    import numpy as np

    # Define some input arrays
    flair = np.random.rand(256, 256)
    t1 = np.random.rand(256, 256)
    label = np.random.randint(0, 3, size=(256, 256))
    # Call the dataAugmentation function
    flairAug, t1Aug, labelAug = dataAugmentation(flair, t1, label)
    # Check that the output arrays have the same shape as the input arrays
    assert flairAug.shape == flair.shape
    assert t1Aug.shape == t1.shape
    assert labelAug.shape == label.shape
    # Check that the output arrays are different from the input arrays
    assert not np.array_equal(flairAug, flair)
    assert not np.array_equal(t1Aug, t1)
    assert not np.array_equal(labelAug, label)

def test_scheduler():
    '''
    This function tests if the learning rate returned by the scheduler function is correct for different epochs.
    '''
    from General_Functions.Training_Functions import scheduler
    import tensorflow as tf
    
    # Test initial learning rate
    assert scheduler(0, 0.001) == 0.001
    
    # Test learning rate after 5 epochs
    assert scheduler(5, 0.001) == 0.001
    
    # Test learning rate after 10 epochs
    assert scheduler(10, 0.001) == 0.001 * tf.math.exp(-0.1)
    
    # Test learning rate after 20 epochs
    assert scheduler(20, 0.001) == 0.001 * tf.math.exp(-0.1)
    
    # Test learning rate after 50 epochs
    assert scheduler(50, 0.001) == 0.001 * tf.math.exp(-0.1)


def test_crop_image():
    from General_Functions.image_preprocessing import crop_image
    import numpy as np

    # Create a test image of size 512x512
    test_image = np.zeros((512, 512))

    # Call crop_image function with standard_dimentions = 256
    cropped_image = crop_image(test_image)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (256, 256)

    # Call crop_image function with standard_dimentions = 128
    cropped_image = crop_image(test_image, standard_dimentions=128)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (128, 128)

    # Call crop_image function with an image that has odd dimensions
    test_image = np.zeros((513, 513))
    cropped_image = crop_image(test_image)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (256, 256)


def test_gaussian_normalisation():
    from General_Functions.image_preprocessing import gaussian_normalisation
    import numpy as np

    # Create a test image of size 10x10
    test_image = np.random.rand(10, 10)

    # Create a binary brain mask with a hole in the middle
    brain_mask = np.zeros((10, 10))
    brain_mask[2:8, 2:8] = 1.0
    brain_mask[4:6, 4:6] = 0.0

    # Call gaussian_normalisation function
    normalised_image = gaussian_normalisation(test_image, brain_mask)

    # Check if the output image has the correct shape
    assert normalised_image.shape == (10, 10)

    # Check if the output image has zero mean and unit variance within the brain mask
    assert np.isclose(np.mean(normalised_image[brain_mask == 1.0]), 0.0, rtol=1e-3)
    assert np.isclose(np.std(normalised_image[brain_mask == 1.0]), 1.0, rtol=1e-3)


def test_float32_converter():
    from General_Functions.image_preprocessing import float32_converter
    import numpy as np

    # Create a test image of size 2x2
    test_image = np.array([[1, 2], [3, 4]])

    # Call the float32_converter function
    float32_image = float32_converter(test_image)

    # Check if the output image has the correct type
    assert float32_image.dtype == np.float32

    # Check if the output image has the correct shape and values
    assert np.allclose(float32_image, np.float32([[1, 2], [3, 4]]))
    

def test_imagePreProcessing():
    from General_Functions.image_preprocessing import imagePreProcessing, float32_converter, gaussian_normalisation, crop_image
    import numpy as np
    # Create a test image of size 512x512 and a corresponding brain mask and label
    test_image = np.random.rand(512, 512)
    test_brain_mask = np.random.randint(2, size=(512, 512))
    test_label = np.random.randint(3, size=(512, 512))

    # Call the imagePreProcessing function
    preprocessed_image, preprocessed_label = imagePreProcessing(test_image, test_brain_mask, test_label)

    # Check if the output images have the correct type
    assert preprocessed_image.dtype == np.float32
    assert preprocessed_label.dtype == np.float32

    # Check if the output images have the correct shape
    assert preprocessed_image.shape == (256, 256)
    assert preprocessed_label.shape == (256, 256)

    # Check if the output images have the correct values
    assert np.allclose(np.mean(preprocessed_image[crop_image(test_brain_mask) == 1.0]), 0.0, rtol=1e-7, atol=1e-7)
    assert np.allclose(np.std(preprocessed_image[crop_image(test_brain_mask) == 1.0]), 1.0, rtol=1e-7, atol=1e-7)
    assert np.allclose(preprocessed_label, crop_image(test_label))
