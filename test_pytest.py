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
        assert np.array_equal(img, dummy_img)


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
        
    saved_image = readImage(os.path.join(img_path, f'{fname}.nii'))

    assert np.array_equal(dummy_img, saved_image)


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
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])
    expected_output = np.array([[[1., 5.], [2., 6.]], [[3., 7.], [4., 8.]]], dtype=np.float32)
    concatenated_img = concatenateImages(flair_img, t1w_img)

    # Test successful concatenation
    assert np.allclose(concatenated_img, expected_output)

def test_concatenateImages_shape():
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])
    concatenated_img = concatenateImages(flair_img, t1w_img)

    # Check that the shape of the concatenated image is correct
    assert concatenated_img.shape == (2, 2, 2)

def test_concatenateImages_matchWithOriginalImages():
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])
    concatenated_img = concatenateImages(flair_img, t1w_img)

    # Check that the first channel of the concatenated image matches the first input image
    assert np.array_equal(concatenated_img[..., 0], flair_img)

    # Check that the second channel of the concatenated image matches the second input image
    assert np.array_equal(concatenated_img[..., 1], t1w_img)


def test_concatenateImages_inputNotNumpy():
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
    from General_Functions.Training_Functions import scheduler
    import tensorflow as tf
    
    # Test learning rate stays the same before epoch 10
    for epoch in range(10):
        assert scheduler(epoch, 0.1) == 0.1


def test_scheduler_learningRateDecreases():
    from General_Functions.Training_Functions import scheduler
    import tensorflow as tf

    # Test learning rate decreases after epoch 10 with the exponential decay formula
    learning_rate = 0.1
    for epoch in range(10, 20):
        learning_rate = scheduler(epoch, learning_rate)

        assert abs(learning_rate - (0.1 * tf.math.exp(-0.1 * (epoch - 9)))) < 1e-7


def test_crop_image_biggerImage():
    from General_Functions.image_preprocessing import crop_image
    import numpy as np

    # Create a test image of size 512x512
    test_image = np.zeros((512, 512))

    # Call crop_image function with standard_dimentions = 256
    cropped_image = crop_image(test_image)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (256, 256)


def test_crop_image_smallerCrop():
    from General_Functions.image_preprocessing import crop_image
    import numpy as np

    # Create a test image of size 512x512
    test_image = np.zeros((512, 512))

    # Call crop_image function with standard_dimentions = 128
    cropped_image = crop_image(test_image, standard_dimensions=128)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (128, 128)

def test_crop_image_oddDimentions():
    from General_Functions.image_preprocessing import crop_image
    import numpy as np

    # Call crop_image function with an image that has odd dimensions
    test_image = np.zeros((513, 513))
    cropped_image = crop_image(test_image)

    # Check if the output image has the correct shape
    assert cropped_image.shape == (256, 256)


def test_gaussian_normalisation_shape():
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


def test_gaussian_normalisation_mean_std():
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
