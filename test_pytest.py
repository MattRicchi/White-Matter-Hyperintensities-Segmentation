#!/usr/bin/env python3
'''
This script contains all the test functions used to test the code

Author: Mattia Ricchi
Date: May 2023
'''

def test_readImage_read():
    '''
    This tests that the readImage function correctly reads and opens the desired NIfTI image file.

    GIVEN: the file path of a medical image to be read.
    WHEN: the readImage function is called with the path to the medical image file.
    THEN: the function returns the contents of the image file as a NumPy array.
    '''
    from General_Functions.Nii_Functions import readImage
    import numpy as np
    from tempfile import TemporaryDirectory
    from pathlib import Path
    import nibabel as nib

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
    from General_Functions.Nii_Functions import readImage
    import pytest

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
    from General_Functions.Nii_Functions import readImage
    import pytest
    import numpy as np
    from tempfile import TemporaryDirectory
    from pathlib import Path
    import nibabel as nib

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
    from General_Functions.Nii_Functions import saveSlice, readImage
    import numpy as np
    import os
    from pathlib import Path

    img_path = os.path.join(os.getcwd(), 'test_folder')
    fname = 'test_image'
    dummy_img = np.random.rand(10, 10)
        
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
    from General_Functions.Nii_Functions import saveSlice
    import tempfile
    import numpy as np
    import pytest

    # Generate test data
    img = np.random.rand(10, 10)
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
    from General_Functions.Nii_Functions import saveSlice
    import numpy as np
    import pytest

    # Generate test data
    img = np.random.rand(10, 10)
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
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])
    expected_output = np.array([[[1., 5.], [2., 6.]], [[3., 7.], [4., 8.]]], dtype=np.float32)
    concatenated_img = concatenateImages(flair_img, t1w_img)

    # Test successful concatenation
    assert np.allclose(concatenated_img, expected_output), "The concatenated image is not equale to the expected one."

def test_concatenateImages_shape():
    '''
    This is to test that the concatenateImages function correctly concatenates the Flair and T1W images along the third axis into a 3D image.

    GIVEN: a 2D flair and a 2D t1w images of dimensions (x, y)
    WHEN: the concatenateImages is applied to the flair and the t1w images
    THEN: the function returns a 3D image of dimensions (x, y, 2)
    '''
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])
    concatenated_img = concatenateImages(flair_img, t1w_img)

    # Check that the shape of the concatenated image is correct
    assert concatenated_img.shape == (2, 2, 2), "The shape of the concatenated image is incorrect."

def test_concatenateImages_matchWithOriginalImages():
    '''
    This is to test that the concatenateImages function correctly concatenates the Flair and T1W images along the third axis into a 3D image.

    GIVEN: a 2D flair and a 2D t1w images
    WHEN: the concatenateImages is applied to the flair and the t1w images
    THEN: the function returns a 3D image with on the first axis the flair image and on the second axis the t1w image
    '''
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np

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
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np
    import pytest

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
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np
    import pytest

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
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np
    import pytest

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
    from General_Functions.Training_Functions import dataAugmentation
    import numpy as np

    # Define some input arrays
    flair = np.random.rand(256, 256)
    t1 = np.random.rand(256, 256)
    label = np.random.randint(0, 2, size=(256, 256))

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
    from General_Functions.Training_Functions import dataAugmentation
    import numpy as np

    # Define some input arrays
    flair = np.random.rand(256, 256)
    t1 = np.random.rand(256, 256)
    label = np.random.randint(0, 2, size=(256, 256))

    # Call the dataAugmentation function
    flairAug, t1Aug, labelAug = dataAugmentation(flair, t1, label)
    
    # Check that the output arrays are different from the input arrays
    assert not np.array_equal(flairAug, flair), "Output FLAIR image is identical to input."
    assert not np.array_equal(t1Aug, t1), "Output T1 image is identical to input."
    assert not np.array_equal(labelAug, label), "Output label image is identical to input."


def test_scheduler_first10epochs():
    '''
    This is to test that the scheduler function leaves the learning rate unchanged for the first ten epochs.
    
    GIVEN: the epoch number smaller than 10 and the initial learning rate
    WHEN: the scheduler function is called 
    THEN: the learning rate remains unchanged 
    '''
    from General_Functions.Training_Functions import scheduler
    import tensorflow as tf
    
    # Test learning rate stays the same before epoch 10
    for epoch in range(10):
        assert scheduler(epoch, 0.1) == 0.1, "Learning rate is not constant for the first 10 epochs."


def test_scheduler_learningRateDecreases():
    '''
    This is to test that the scheduler function causes the learning rate to decrease exponentially after the first ten epochs.
    
    GIVEN: the epoch number grater than 10 and the initial learning rate
    WHEN: the scheduler function is called 
    THEN: the learning rate decreases exponentially
    '''
    from General_Functions.Training_Functions import scheduler
    import tensorflow as tf

    # Test learning rate decreases after epoch 10 with the exponential decay formula
    learning_rate = 0.1
    for epoch in range(10, 20):
        learning_rate = scheduler(epoch, learning_rate)

        assert abs(learning_rate - (0.1 * tf.math.exp(-0.1 * (epoch - 9)))) < 1e-7, "Learning rate doesn't decrease exponentially."


def test_crop_image_cropShape():
    '''
    This is to test that the crop_image function correctly crops the input images to the dimentions (256, 256).

    GIVEN: 2D input image
    WHEN: the crop_image function is applied to the image
    THEN: the function returns the input image cropped to have shape (256, 256)
    '''
    from General_Functions.image_preprocessing import crop_image
    import numpy as np

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
    from General_Functions.image_preprocessing import crop_image
    import numpy as np

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
    from General_Functions.image_preprocessing import crop_image
    import numpy as np

    # Call crop_image function with an image that has odd dimensions
    test_image = np.zeros((513, 513))
    cropped_image = crop_image(test_image)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (256, 256), "Crop shape doesn't match (256, 256)."


def test_gaussian_normalisation_shape():
    '''
    This is to test that the gaussian_normalisation function returns an image with the same shape as the input.

    GIVEN: a 2D image to be normalised and its brain_mask image
    WHEN: the gaussian_normalisation is applied to the inputs
    THEN: the function returns an image with the same shape as the input one
    '''
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
    assert normalised_image.shape == (10, 10), "Normalised image shape doesn't match the shape of the original image."


def test_gaussian_normalisation_mean_std():
    '''
    This is to test that the gaussian_normalisation function returns an image with zero mean and unit standard deviation.

    GIVEN: a 2D image to be normalised and its brain_mask image
    WHEN: the gaussian_normalisation is applied to the inputs
    THEN: the function returns an image with mean = 0 and standard deviation = 1
    '''
    from General_Functions.image_preprocessing import gaussian_normalisation
    import numpy as np

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