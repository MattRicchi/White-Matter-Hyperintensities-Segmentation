'''
This script contains all the test functions used to test the code

Author: Mattia Ricchi
Date: May 2023
'''
#!/usr/bin/env python3

def test_readImage():
    from General_Functions.Nii_Functions import readImage
    import pytest
    import numpy as np
    from numpy.testing import assert_array_equal
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
        assert_array_equal(img, dummy_img)
        
        # Test that a FileNotFoundError is raised if the image path does not exist
        with pytest.raises(FileNotFoundError):
            readImage('nonexistent.nii.gz')
            
        # Test that a nibabel.filebasedimages.ImageFileError is raised if the file path is not a valid medical image file
        with open(str(img_path), 'w') as f:
            f.write('not a medical image file')
        with pytest.raises(nib.filebasedimages.ImageFileError):
            readImage(str(img_path))


def test_saveSlice():
    from General_Functions.Nii_Functions import saveSlice
    import os
    import tempfile
    import numpy as np
    import nibabel as nib
    import pytest

    # Generate test data
    img = np.random.rand(10, 10, 10)
    fname = "test_image"
    
    # Test saving to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        saveSlice(img, fname, temp_dir)
        assert os.path.exists(os.path.join(temp_dir, f"{fname}.nii"))

    # Test saving with invalid inputs
    with pytest.raises(TypeError):
        saveSlice("invalid_input", fname, temp_dir)
    
    with pytest.raises(ValueError):
        saveSlice(img, "", temp_dir)
    
    with pytest.raises(ValueError):
        saveSlice(img, fname, "")

    # Test saving to a directory that cannot be created
    with pytest.raises(OSError):
        saveSlice(img, fname, "/path/that/does/not/exist")


def test_concatenateImages():
    from General_Functions.Nii_Functions import concatenateImages
    import numpy as np
    import pytest

    # Create two test images
    flair_img = np.array([[1, 2], [3, 4]])
    t1w_img = np.array([[5, 6], [7, 8]])
    expected_output = np.array([[[1., 5.], [2., 6.]], [[3., 7.], [4., 8.]]], dtype=np.float32)
    concatenated_img = concatenateImages(flair_img, t1w_img)

    # Test successful concatenation
    assert np.allclose(concatenated_img, expected_output)

    # Check that the shape of the concatenated image is correct
    assert concatenated_img.shape == (2, 2, 2)

    # Check that the first channel of the concatenated image matches the first input image
    assert np.array_equal(concatenated_img[..., 0], flair_img)

    # Check that the second channel of the concatenated image matches the second input image
    assert np.array_equal(concatenated_img[..., 1], t1w_img)

    # Test that the function raises a TypeError if the inputs are not numpy arrays
    with pytest.raises(TypeError):
        concatenateImages(list(flair_img), t1w_img)

    with pytest.raises(TypeError):
        concatenateImages(flair_img, 3)

    # Test that the function raises a ValueError if the inputs do not have the same shape
    with pytest.raises(ValueError):
        concatenateImages(flair_img, np.array([[5, 6], [7, 8], [9, 10]]))

    # Test that the function raises a ValueError if either input is not a 2D numpy array
    with pytest.raises(ValueError):
        concatenateImages(flair_img[0], t1w_img)

    with pytest.raises(ValueError):
        concatenateImages(flair_img, t1w_img[..., np.newaxis])


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
    assert flairAug.shape == flair.shape, "Output FLAIR image shape is incorrect."
    assert t1Aug.shape == t1.shape, "Output T1 image shape is incorrect."
    assert labelAug.shape == label.shape, "Output label image shape is incorrect."
    
    # Check that the output arrays are different from the input arrays
    assert not np.array_equal(flairAug, flair), "Output FLAIR image is identical to input."
    assert not np.array_equal(t1Aug, t1), "Output T1 image is identical to input."
    assert not np.array_equal(labelAug, label), "Output label image is identical to input."

def test_learning_rate_scheduler():
    from General_Functions.Training_Functions import learning_rate_scheduler
    import tensorflow as tf

    # Test case 1: learning rate stays the same before epoch 10
    assert learning_rate_scheduler(5, 0.1) == 0.1

    # Test case 2: learning rate decreases after epoch 10
    assert learning_rate_scheduler(10, 0.1) == 0.1 * tf.math.exp(-0.1)
    assert learning_rate_scheduler(11, 0.1 * tf.math.exp(-0.1)) == 0.1 * tf.math.exp(-0.2)

    # Test case 3: learning rate decreases with the exponential decay formula
#    for epoch in range(10, 20):
#        lr = 0.1 * tf.math.exp(-0.1 * (epoch - 9))
#        assert abs(learning_rate_scheduler(epoch, 0.1) - lr) < 1e-7

    # Test case 4: learning rate decreases for a larger initial learning rate
#    assert abs(learning_rate_scheduler(15, 0.5) - (0.5 * tf.math.exp(-0.5))) < 1e-7

    # Test case 5: function raises no exceptions for valid inputs
    try:
        learning_rate_scheduler(5, 0.1)
        learning_rate_scheduler(10, 0.1)
        learning_rate_scheduler(15, 0.5)
    except:
        assert False

    # Test case 6: function raises a TypeError for invalid input types
    try:
        learning_rate_scheduler('epoch', 0.1)
        assert False
    except TypeError:
        pass



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
