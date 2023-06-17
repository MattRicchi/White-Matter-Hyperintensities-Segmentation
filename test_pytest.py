#!/usr/bin/env python3
'''
This script contains all the test functions used to test the code

Author: Mattia Ricchi
Date: May 2023
'''
import numpy as np
import tensorflow as tf
import tempfile
from tempfile import TemporaryDirectory
from pathlib import Path
import nibabel as nib
import pytest
import os

from General_Functions.Nii_Functions import readImage, saveSlice, concatenateImages
from General_Functions.Training_Functions import dataAugmentation, scheduler, get_test_patients, has_brain, add_to_test_data, add_to_train_data, build_train_test_data
from General_Functions.image_preprocessing import crop_image, gaussian_normalisation, imagePreProcessing

def test_readImage_read():
    '''
    This tests that the readImage function correctly reads and opens the desired NIfTI image file.

    GIVEN: the file path of a medical image to be read.
    WHEN: the readImage function is called with the path to the medical image file.
    THEN: the function returns the contents of the image file as a NumPy array.
    '''

    # Create a temporary directory and a dummy image file
    with TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / 'test.nii.gz'
        dummy_img = np.array([[[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]], [[0.77, 0.88, 0.99], [1.11, 1.22, 1.33]]])
        nib.save(nib.Nifti1Image(dummy_img, np.eye(4)), str(img_path))
        
        # Test reading the dummy image
        img = readImage(str(img_path))
        assert np.array_equal(img, dummy_img), "The read image is not equal to the original image."


def test_readImage_FileNotFoundError():
    '''
    This tests that the readImage function correctly raises a FileNotFoundError when given a path to a non-existent file.
    
    GIVEN: a non-existent file path
    WHEN: the readImage function is called with the path to the non-existent file
    THEN: the function raises a FileNotFoundError
    '''

    # Test that a FileNotFoundError is raised if the image path does not exist
    with pytest.raises(FileNotFoundError):
        readImage('nonexistent.nii.gz')


def test_readImage_ImageFileError():
    '''
    This is to test that the readImage function raises an ImageFileError when given a not valid NIfTI image file as input.
    
    GIVEN: a file path that points to a file that is not a valid NIfTI image file
    WHEN: the readImage function is called with the path to the invalid file
    THEN: the function raises a nibabel.filebasedimages.ImageFileError
    '''

    # Create a temporary directory and a dummy image file
    with TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / 'test.nii.gz'
        dummy_img = np.array([[[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]], [[0.77, 0.88, 0.99], [1.11, 1.22, 1.33]]])
        nib.save(nib.Nifti1Image(dummy_img, np.eye(4)), str(img_path))
            
        # Test that a nibabel.filebasedimages.ImageFileError is raised if the file path is not a valid medical image file
        with open(str(img_path), 'w') as f:
            f.write('not a medical image file')
        with pytest.raises(nib.filebasedimages.ImageFileError):
            readImage(str(img_path))


def test_saveSlice_sliceSavedCorrectly():
    '''
    This is to test that the saveSlice function correctly saves the image given in input as a NIfTI file. 
    
    GIVEN: an image to be saved, a name to save the image with and the path where to save the image
    WHEN: the saveSlice function is called with the correct inputs 
    THEN: the function saves the image as NIfTI file
    '''

    img_path = os.path.join(os.getcwd(), 'test_folder')
    fname = 'test_image'
    dummy_img = np.array([[[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]], [[0.77, 0.88, 0.99], [1.11, 1.22, 1.33]]])
        
    saveSlice(dummy_img, fname, img_path)
        
    saved_image = readImage(os.path.join(img_path, f'{fname}.nii')) # The readImage function is tested separately

    assert np.array_equal(dummy_img, saved_image), "The saved image is not equal to the original one."


def test_saveSlice_invalidInputs():
    '''
    This is to test that the saveSlice function raises correct errors when given the wrong inputs. 
    
    GIVEN: wrong input types, i.e., string as input image and empy string as file name or directory
    WHEN: the saveSlice function is called with the wrong inputs 
    THEN: the function raises the correct error, TypeError or ValueError
    '''

    # Generate test data
    img = np.array([[[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]], [[0.77, 0.88, 0.99], [1.11, 1.22, 1.33]]])
    fname = "test_image"

    with tempfile.TemporaryDirectory() as temp_dir:

        # Test saving with invalid image 
        with pytest.raises(TypeError):
            saveSlice("invalid_input", fname, temp_dir)

        # Test saving with invalid image name
        with pytest.raises(ValueError):
            saveSlice(img, "", temp_dir)

        # Test saving with invalid output directory
        with pytest.raises(ValueError):
            saveSlice(img, fname, "")

def test_saveSlice_nonValid_Dir():
    '''
    This is to test that the saveSlice function raises OSError when the output directory does not exists and cannot be created. 
    
    GIVEN: a path that does not exists and cannot be created 
    WHEN: the saveSlice function is called with the wrong path 
    THEN: the function raises the correct OSError
    '''

    # Generate test data
    img = np.array([[[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]], [[0.77, 0.88, 0.99], [1.11, 1.22, 1.33]]])
    fname = "test_image"

    # Test saving to a directory that cannot be created
    with pytest.raises(OSError):
        saveSlice(img, fname, "/path/that/does/not/exist")


def test_concatenateImages_correctConcatenation():
    '''
    This is to test that the concatenateImages function correctly concatenates the Flair and T1W images along the third axis into a 3D image.

    GIVEN: a 2D flair and a 2D t1w images
    WHEN: the concatenateImages is applied to the flair and the t1w images
    THEN: the function returns a 3D image given by the concatenation of the flair and t1w images along the third axis
    '''
    
    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])
    expected_output = np.array([[[1., 5.], [2., 6.]], [[3., 7.], [4., 8.]]], dtype=np.float32)
    concatenated_img = concatenateImages(flair_img, t1w_img)

    # Test successful concatenation
    assert np.allclose(concatenated_img, expected_output), "The concatenated image is not equale to the expected one."

def test_concatenateImages_matchWithOriginalImages():
    '''
    This is to test that the concatenateImages function correctly concatenates the Flair and T1W images along the third axis into a 3D image.

    GIVEN: a 2D flair and a 2D t1w images
    WHEN: the concatenateImages is applied to the flair and the t1w images
    THEN: the function returns a 3D image with on the first axis the flair image and on the second axis the t1w image
    '''

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])
    concatenated_img = concatenateImages(flair_img, t1w_img)

    # Check that the first channel of the concatenated image matches the first input image
    assert np.array_equal(concatenated_img[..., 0], flair_img), "The first channel of the concatenated image doesn't match the first input image."

    # Check that the second channel of the concatenated image matches the second input image
    assert np.array_equal(concatenated_img[..., 1], t1w_img), "The second channel of the concatenated image doesn't match the second input image."


def test_concatenateImages_inputNotNumpy():
    '''
    This is to test that the concatenateImages raises a TypeError when the images in input are not numpy arrays

    GIVEN: flair and t1w images but either one of the two is not a numpy array
    WHEN: the concatenateImages function is applied to the flair and t1w images
    THEN: the function returns a TypeError 
    '''

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])

    # Test that the function raises a TypeError if the inputs are not numpy arrays
    with pytest.raises(TypeError):
        concatenateImages(list(flair_img), t1w_img)

    with pytest.raises(TypeError):
        concatenateImages(flair_img, 3)


def test_concatenateImages_inputsNotSameShape():
    '''
    This is to test that the concatenateImages raises a ValueError if the flair and t1w images do not have the same shape

    GIVEN: flair and t1w images with different shapes 
    WHEN: the concatenateImages is applied to them
    THEN: the function returns a ValueError
    '''

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8], [9, 10]])

    # Test that the function raises a ValueError if the inputs do not have the same shape
    with pytest.raises(ValueError):
        concatenateImages(flair_img, t1w_img)


def test_concatenateImages_inputsNot2D():
    '''
    This is to test that the concatenateImages raises a ValueError when the images in input are not 2D

    GIVEN: flair and t1w images but either one of the two is not a 2D image
    WHEN: the concatenateImages function is applied to the flair and t1w images
    THEN: the function returns a ValueError
    '''

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])

    # Test that the function raises a ValueError if either input is not a 2D numpy array
    with pytest.raises(ValueError):
        concatenateImages(flair_img[0], t1w_img)

    with pytest.raises(ValueError):
        concatenateImages(flair_img, t1w_img[..., np.newaxis])


def test_dataAugmentation_shape():
    '''
    This is to test that the dataAugmentation function returns flair, t1w and label images with the same shape as the original ones.

    GIVEN: flair image, t1w image and the label image of the same slice
    WHEN: the dataAugmentation function is applied to the images
    THEN: the function returns new flair, t1w and label images with the same shape as the originals
    '''
    
    # Create a flair image of size 10x10 with a square of different gray levels in the middle
    flair = np.zeros((10, 10))
    flair[2:4, 2:6] = 1.0
    flair[4:8, 2:4] = 2.0
    flair[6:8, 4:8] = 3.0
    flair[2:6, 6:8] = 4.0
    
    # Create a t1 image of size 10x10 with a square of different gray levels in the middle
    t1 = np.zeros((10, 10))
    t1[2:4, 2:6] = 1.0
    t1[4:8, 2:4] = 2.0
    t1[6:8, 4:8] = 3.0
    t1[2:6, 6:8] = 4.0
    
    # Create the label mask with a hole in the middle
    label = np.zeros((10, 10))
    label[2:8, 2:8] = 1.0
    label[4:6, 4:6] = 0.0

    # Call the dataAugmentation function
    flairAug, t1Aug, labelAug = dataAugmentation(flair, t1, label)

    # Check that the output arrays have the same shape as the input arrays
    assert flairAug.shape == flair.shape, "Output FLAIR image shape is incorrect."
    assert t1Aug.shape == t1.shape, "Output T1 image shape is incorrect."
    assert labelAug.shape == label.shape, "Output label image shape is incorrect."


def test_dataAugmentation_AugmentedDifferentFromInputs():
    '''
    This is to test that the dataAugmentation function returns flair, t1w and label new images.

    GIVEN: flair image, t1w image and the label image of the same slice
    WHEN: the dataAugmentation function is applied to the images
    THEN: the function returns new flair, t1w and label images
    '''
    
    # Create a flair image of size 10x10 with a square of different gray levels in the middle
    flair = np.zeros((10, 10))
    flair[2:4, 2:6] = 1.0
    flair[4:8, 2:4] = 2.0
    flair[6:8, 4:8] = 3.0
    flair[2:6, 6:8] = 4.0
    
    # Create a t1 image of size 10x10 with a square of different gray levels in the middle
    t1 = np.zeros((10, 10))
    t1[2:4, 2:6] = 1.0
    t1[4:8, 2:4] = 2.0
    t1[6:8, 4:8] = 3.0
    t1[2:6, 6:8] = 4.0
    
    # Create the label mask with a hole in the middle
    label = np.zeros((10, 10))
    label[2:8, 2:8] = 1.0
    label[4:6, 4:6] = 0.0

    # Call the dataAugmentation function
    flairAug, t1Aug, labelAug = dataAugmentation(flair, t1, label)
    
    # Check that the output arrays are different from the input arrays
    assert not np.array_equal(flairAug, flair), "Output FLAIR image is identical to input."
    assert not np.array_equal(t1Aug, t1), "Output T1 image is identical to input."
    assert not np.array_equal(labelAug, label), "Output label image is identical to input."


def test_learning_rate_scheduler():
    '''
    This is to test that the scheduler function leaves the learning rate unchanged for the first ten epochs and then it decreases exponentially.
    
    GIVEN: the initial learning rate
    WHEN: the scheduler function is called 
    THEN: the learning rate remains unchanged for the first ten epochs and then it decreases exponentially
    '''
    learning_rate = 0.1
    expected_learning_rates = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09048374742269516,
                               0.0818730816245079, 0.07408183068037033, 0.06703201681375504,
                               0.06065307930111885, 0.054881177842617035, 0.04965854436159134,
                               0.044932909309864044, 0.04065697640180588, 0.03678795322775841,
                               0.03328711539506912]

    for epoch, expected_lr in enumerate(expected_learning_rates, start=1):
        learning_rate = scheduler(epoch, learning_rate)
        assert learning_rate == expected_lr, f"Learning rate mismatch at epoch {epoch}. Expected: {expected_lr}, Got: {learning_rate}"


def test_crop_image_cropShape():
    '''
    This is to test that the crop_image function correctly crops the input images to the dimentions (256, 256).

    GIVEN: 2D input image
    WHEN: the crop_image function is applied to the image
    THEN: the function returns the input image cropped to have shape (256, 256)
    '''

    # Create a test image of size 512x512
    test_image = np.zeros((512, 512))

    # Call crop_image function
    cropped_image = crop_image(test_image)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (256, 256), "Crop shape dosen't match (256, 256)."


def test_crop_image_smallerCrop():
    '''
    This is to test that the crop_image function correctly crops the input images to the desired dimentions.

    GIVEN: 2D input image
    WHEN: the crop_image function is applied to the image
    THEN: the function returns the input image cropped to have the wanted shape
    '''

    # Create a test image of size 512x512
    test_image = np.zeros((512, 512))

    # Call crop_image function with standard_dimensions = 128
    cropped_image = crop_image(test_image, standard_dimensions=128)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (128, 128), "Crop shape doesn't match the requested one."


def test_crop_image_oddDimensions():
    '''
    This is to test that the crop_image function also works correctly with an input image with odd dimensions.

    GIVEN: input image with odd dimensions
    WHEN: crop_image function is applied to the input image
    THEN: the function correctly crops the image to have shape equal to (256, 256)
    '''

    # Call crop_image function with an image that has odd dimensions
    test_image = np.zeros((513, 513))
    cropped_image = crop_image(test_image)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (256, 256), "Crop shape doesn't match (256, 256)."


def test_crop_image_values():
    '''
    This is to test that the crop_image function returns an image with the correct values.

    GIVEN: input image
    WHEN: crop_image function is applied to the input image
    THEN: the cropped image has the corrected values
    '''
    
    # Create a test image with known values
    image = np.array([[1, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12],
                      [13, 14, 15, 16, 17, 18],
                      [19, 20, 21, 22, 23, 24],
                      [25, 26, 27, 28, 29, 30],
                      [31, 32, 33, 34, 35, 36]])

    # Define the expected cropped image dimensions
    expected_dimensions = 4

    # Call the function to get the actual cropped image
    actual_cropped_image = crop_image(image, expected_dimensions)

    # Define the manually cropped region from the original image
    manual_cropped_region = np.array([[8, 9, 10, 11],
                                      [14, 15, 16, 17],
                                      [20, 21, 22, 23],
                                      [26, 27, 28, 29]])

    # Check if the content of the cropped image matches the manually cropped region
    assert np.array_equal(actual_cropped_image, manual_cropped_region), "Content of cropped image does not match the manually cropped region."


def test_gaussian_normalisation_values():
    '''
    This is to test that the gaussian_normalisation function returns an image with the expected values.

    GIVEN: a 2D image to be normalised and its brain_mask image
    WHEN: the gaussian_normalisation is applied to the inputs
    THEN: the function returns an image with normalised gray level intensity
    '''

    # Define test image and brain mask
    image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    brain_mask = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])

    # Define expected normalized image
    expected_result = np.array([[-3.22047024, -2.6349302, -2.04939015],
                               [-1.46385011, -0.87831007, -0.29277002],
                               [0.29277002, 0.87831007, 1.46385011]])

    # Call the function to get the actual result
    actual_result = gaussian_normalisation(image, brain_mask)

    # Check if the actual result matches the expected result
    assert np.allclose(actual_result, expected_result), f"Normalization incorrect. Expected:\n{expected_result}\nActual:\n{actual_result}"
    

def test_gaussian_normalisation_mean_std():
    '''
    This is to test that the gaussian_normalisation function returns an image with zero mean and unit standard deviation.

    GIVEN: a 2D image to be normalised and its brain_mask image
    WHEN: the gaussian_normalisation is applied to the inputs
    THEN: the function returns an image with mean = 0 and standard deviation = 1
    '''

    # Create a test image of size 10x10 with a square of different gray levels in the middle
    test_image = np.zeros((10, 10))
    test_image[2:4, 2:6] = 1.0
    test_image[4:8, 2:4] = 2.0
    test_image[6:8, 4:8] = 3.0
    test_image[2:6, 6:8] = 4.0

    # Create the binary mask with a hole in the middle
    brain_mask = np.zeros((10, 10))
    brain_mask[2:8, 2:8] = 1.0
    brain_mask[4:6, 4:6] = 0.0

    # Call gaussian_normalisation function
    normalised_image = gaussian_normalisation(test_image, brain_mask)

    # Check if the output image has zero mean and unit variance within the brain mask
    assert np.isclose(np.mean(normalised_image[brain_mask == 1.0]), 0.0, rtol=1e-3), "The normalised image mean is not close to 0."
    assert np.isclose(np.std(normalised_image[brain_mask == 1.0]), 1.0, rtol=1e-3), "the normalised image standard deviation is not close to 1."
    
def test_get_test_patients():
    '''
    This is to test that the get_test_patients correctly reads the test_patients.txt and builds an array with the content of the txt file.
    
    GIVEN: a txt file containing the numbers corresponding to test patients
    WHEN: the get_test_patients is called
    THEN: the content of the txt file is correctly used to build the test_patients array
    '''
    
    # Set the path to the test patients file
    test_patients_file = "test_folder/test_patients.txt"

    # Call the get_test_patients function
    test_patients = get_test_patients(test_patients_file)

    # Define the expected result
    expected_result = np.array([1, 2, 3, 4, 5])
        
    # Assert that the returned value matches the expected result
    np.testing.assert_array_equal(test_patients, expected_result)
    

def test_has_brain_with_brain():
    '''
    This is to test that the has_brain function returns True when a brain mask with non zero values is given.
    
    GIVEN: a brain mask with non zero values
    WHEN: the has_brain function is applied to the brain mask
    THEN: the has_brain returns True
    '''
    
    # Create a brain mask with non-zero values
    brain_mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        
    # Call the has_brain function
    result = has_brain(brain_mask)
        
    # Assert that the result is True
    assert result == True
    
    
def test_has_brain_without_brain():
    '''
    This is to test that the has_brain function returns False when a brain mask with zero values is given.
    
    GIVEN: a brain mask with zero values
    WHEN: the has_brain function is applied to the brain mask
    THEN: the has_brain returns False
    '''
    
    # Create a brain mask with zero values
    brain_mask = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        
    # Call the has_brain function
    result = has_brain(brain_mask)
        
    # Assert that the result is True
    assert result == False
    

def test_add_to_test_data():
    '''
    This is to test that the add_to_test_data function correctly adds the given images to the TEST_IMAGES array.
    
    GIVEN: An image that will be used to test the network
    WHEN: The add_to_test_data is applied to the image
    THEN: The image is correctly added to the TEST_IMAGES array
    '''
    # Initialize test data arrays
    TEST_IMAGES = np.ndarray((0, 256, 256, 2))
    Image_IDs = np.empty(0)
    
    # Generate sample FLAIR_and_T1W_image and id_
    FLAIR_and_T1W_image_1 = np.ones((256, 256, 2))
    id_1 = 'image001'
    FLAIR_and_T1W_image_2 = np.ones((256, 256, 2)) * 2
    id_2 = 'image002'
    
    # Add the images to test data
    TEST_IMAGES, Image_IDs = add_to_test_data(TEST_IMAGES, Image_IDs, FLAIR_and_T1W_image_1, id_1)
    TEST_IMAGES, Image_IDs = add_to_test_data(TEST_IMAGES, Image_IDs, FLAIR_and_T1W_image_2, id_2)
    
    # Verify the updated arrays after adding the sample images
    assert np.array_equal(TEST_IMAGES[0], FLAIR_and_T1W_image_1)
    assert np.array_equal(Image_IDs[0], id_1)
    assert np.array_equal(TEST_IMAGES[1], FLAIR_and_T1W_image_2)
    assert np.array_equal(Image_IDs, np.array([id_1, id_2]))
    
    
def test_add_to_train_data():
    '''
    This is to test that the add_to_train_data function correctly adds the given images to the TRAIN_IMAGES array, 
    and the correspondent labels to the TRAIN_LABELS array.
    
    GIVEN: An image that will be used to train the network and its gorund truth segmentation map
    WHEN: The add_to_train_data is applied to the image and to the label
    THEN: The image is correctly added to the TRAIN_IMAGES array and the label image to the TRAIN_LABELS array
    '''
    # Initialize train data arrays
    TRAIN_IMAGES = np.ndarray((0, 256, 256, 2))
    TRAIN_LABELS = np.ndarray((0, 256, 256))
    
    # Generate sample FLAIR_and_T1W_image and label
    FLAIR_and_T1W_image_1 = np.ones((256, 256, 2))
    label_image_1 = np.ones((256, 256))
    FLAIR_and_T1W_image_2 = np.ones((256, 256, 2)) * 2
    label_image_2 = np.zeros((256, 256))
    
    # Add the images to train data
    TRAIN_IMAGES, TRAIN_LABELS = add_to_train_data(TRAIN_IMAGES, TRAIN_LABELS, FLAIR_and_T1W_image_1, label_image_1)
    TRAIN_IMAGES, TRAIN_LABELS = add_to_train_data(TRAIN_IMAGES, TRAIN_LABELS, FLAIR_and_T1W_image_2, label_image_2)
    
    # Assert the added data
    assert np.array_equal(TRAIN_IMAGES[0], FLAIR_and_T1W_image_1)
    assert np.array_equal(TRAIN_LABELS[0], label_image_1)
    assert np.array_equal(TRAIN_IMAGES[1], FLAIR_and_T1W_image_2)
    assert np.array_equal(TRAIN_LABELS[1], label_image_2)


def test_build_train_test_data_TEST_IMAGE():
    '''
    This is to test that the build_train_test_data function correctly classifies a given image as a test image
    
    GIVEN: a flair and t1w images whose id corresponds to one of the testing patients
    WHEN: the build_train_test_data function is called
    THEN: the function correctly classifies the image as a test image
    '''
    TRAIN_IMAGES = np.ndarray((0, 256, 256, 2))
    TEST_IMAGES = np.ndarray((0, 256, 256, 2))
    TRAIN_LABELS = np.ndarray((0, 256, 256))
    Image_IDs = np.empty(0)
    
    test_patients = [2, 5]
    
    flair = readImage('test_folder/OnlyBrain/flair/volume-002-004.nii')
    t1w = readImage('test_folder/OnlyBrain/t1w/volume-002-004.nii')
    label = readImage('test_folder/OnlyBrain/label/volume-002-004.nii')
    brain_mask = readImage('test_folder/brain/volume-002-004.nii')
    
    (flair, label) = imagePreProcessing(flair, brain_mask, label)
    (t1w, label) = imagePreProcessing(t1w, brain_mask, label)
    
    FLAIR_and_T1W_image = concatenateImages(flair, t1w)
    
    TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, Image_IDs = build_train_test_data('test_folder/', test_patients, [], 'volume-002-004.nii', TEST_IMAGES, TRAIN_IMAGES, TRAIN_LABELS, Image_IDs)
    
    assert np.array_equal(TEST_IMAGES[0], FLAIR_and_T1W_image)
    assert Image_IDs[0] == 'volume-002-004.nii'
    
    
def test_build_train_test_data_TRAIN_IMAGE():
    '''
    This is to test that the build_train_test_data function correctly classifies a given image as a train image
    
    GIVEN: a flair and t1w images whose id don't correspond to one of the testing patients
    WHEN: the build_train_test_data function is called
    THEN: the function correctly classifies the image as a train image
    '''
    TRAIN_IMAGES = np.ndarray((0, 256, 256, 2))
    TEST_IMAGES = np.ndarray((0, 256, 256, 2))
    TRAIN_LABELS = np.ndarray((0, 256, 256))
    Image_IDs = np.empty(0)
    
    test_patients = [1, 5]
    
    flair = readImage('test_folder/OnlyBrain/flair/volume-002-004.nii')
    t1w = readImage('test_folder/OnlyBrain/t1w/volume-002-004.nii')
    label = readImage('test_folder/OnlyBrain/label/volume-002-004.nii')
    brain_mask = readImage('test_folder/brain/volume-002-004.nii')
    
    (flair, label) = imagePreProcessing(flair, brain_mask, label)
    (t1w, label) = imagePreProcessing(t1w, brain_mask, label)
    
    FLAIR_and_T1W_image = concatenateImages(flair, t1w)
    
    TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, Image_IDs = build_train_test_data('test_folder/', test_patients, [], 'volume-002-004.nii', TEST_IMAGES, TRAIN_IMAGES, TRAIN_LABELS, Image_IDs)
    
    assert np.array_equal(TRAIN_IMAGES[0], FLAIR_and_T1W_image)
    assert np.array_equal(TRAIN_LABELS[0], label)
    
    
def test_build_train_test_data_NO_BRAIN():
    '''
    This is to test that the build_train_test_data function does not update the TRAIN_IMAGES, TEST_IMAGES, TRAIN_LABELS, Image_IDs
    arrays when an image with no brain is given as input.
    
    GIVEN: flair and t1w images that do not contain brain
    WHEN: the build_train_test_data function is called
    THEN: the function correctly does not update the output arrays
    '''
    TRAIN_IMAGES = np.ndarray((0, 256, 256, 2))
    TEST_IMAGES = np.ndarray((0, 256, 256, 2))
    TRAIN_LABELS = np.ndarray((0, 256, 256))
    Image_IDs = np.empty(0)
    
    TRAIN_IMAGES_updated, TRAIN_LABELS_updated, TEST_IMAGES_updated, Image_IDs_updated = build_train_test_data('test_folder/test_NObrain/', [], [], 'volume-002-004.nii', TEST_IMAGES, TRAIN_IMAGES, TRAIN_LABELS, Image_IDs)
    
    assert np.array_equal(TRAIN_IMAGES, TRAIN_IMAGES_updated)
    assert np.array_equal(TRAIN_LABELS, TRAIN_LABELS_updated)
    assert np.array_equal(TEST_IMAGES, TEST_IMAGES_updated)
    assert np.array_equal(Image_IDs, Image_IDs_updated)