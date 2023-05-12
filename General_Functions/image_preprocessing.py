'''
This script contains all the necessary functions for the preprocessing of the MR images. 

Author: Mattia Ricchi
Date: May 2023
'''
def crop_image(image, standard_dimensions=256):
    """
    Crop a given 2D image to a specified size around its center.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D image to be cropped.
    standard_dimensions : int, optional
        The desired size of the cropped image. Default is 256.

    Returns
    -------
    cropped_image : numpy.ndarray
        The cropped image.

    Raises
    ------
    TypeError
        If the input image is not a numpy array.
    ValueError
        If the input image is not a 2D array.
        If the dimensions of the input image are smaller than the specified crop size.

    Notes
    -----
    This function crops the image to the specified size around its center, such that the cropped image has
    the same number of rows and columns as the specified crop size.

    Example
    -------
    >>> import numpy as np
    >>> image = np.random.rand(512, 512)
    >>> cropped_image = crop_image(image, 256)
    >>> cropped_image.shape
    (256, 256)
    """
    import numpy as np

    (image_rows, image_columns) = image.shape

    # Check inputs for correctness
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")
    if not image.ndim == 2:
        raise ValueError("Input image must be a 2D array.")

    # Check that the dimensions of the input image are not smaller than the crop size
    if image_rows < standard_dimensions or image_columns < standard_dimensions:
        raise ValueError("Image dimensions are smaller than the specified crop size.")

    # Crop the input image to have dimensions match the specified crop size
    cropped_image = image[(int(image_rows/2) - int(standard_dimensions/2)) : (int(image_rows/2) + int(standard_dimensions/2)),
              (int(image_columns/2) - int(standard_dimensions/2)) : (int(image_columns/2) + int(standard_dimensions/2))]

    return cropped_image


def gaussian_normalisation(image, brain_mask):
    """
    Apply Gaussian normalization to an image using the mean and standard deviation within a brain mask.

    Parameters:
    ----------
    image : np.ndarray
        The input image to be normalized.
    brain_mask : np.ndarray
        A binary mask indicating the region of the brain in the image.

    Returns:
    -------
    np.ndarray
        The normalized image.

    Raises:
    -------
    TypeError 
        If either image or brain_mask is not a numpy.ndarray.
    ValueError
        If the shapes of the image and brain_mask arrays do not match.

    Notes:
    -----
    Gaussian normalization is a technique commonly used in medical imaging to adjust the intensity of images to have a
    zero mean and unit variance. This is done to reduce the impact of intensity variations due to scanner artifacts
    and other factors.

    Example:
    -------
    >>> image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> brain_mask = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    >>> gaussian_normalisation(image, brain_mask)
    array([[-1.22474487, -1.22474487, -1.22474487],
           [ 0.        ,  0.        ,  0.        ],
           [ 1.22474487,  1.22474487,  1.22474487]])
    """
    
    import numpy as np
    
    # Check inputs for correctness
    if not isinstance(image, np.ndarray) or not isinstance(brain_mask, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")

    # Check that the shape of the image and brain_mask match
    if image.shape != brain_mask.shape:
        raise ValueError("The shapes of the input image and brain mask do not match.")
    
    # Compute the mean and standard deviation of the image within the brain mask
    image_mean = np.mean(image[brain_mask == 1.0])
    image_std = np.std(image[brain_mask == 1.0])

    # Normalize the image using a Gaussian distribution with zero mean and unit variance
    normalised_image = (image - image_mean) / image_std
    
    return normalised_image



def float32_converter(image):
    """
    Converts the input image to a float32 numpy array.

    Parameters
    ----------
    image : ndarray
        Input image as a numpy array.

    Returns
    -------
    ndarray
        Float32 numpy array.

    Raises
    ------
    TypeError
        If the input image is not a numpy array.

    Notes
    -----
    The float32 numpy array is useful for numerical calculations as it provides greater precision than other data types.
    If the input image is already a float32 numpy array, the function returns the same array without any conversion.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[1, 2], [3, 4]])
    >>> float_img = float32_converter(img)
    >>> float_img
    array([[1., 2.],
           [3., 4.]], dtype=float32)
    """
    import numpy as np

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")

    image_float32 = np.float32(image)

    return image_float32



def imagePreProcessing(image, brain_mask, label):
    """
    Applies a series of preprocessing steps to an input image, brain mask, and label.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image to be preprocessed.
    brain_mask : numpy.ndarray
        A mask of the brain volume within the input image.
    label : numpy.ndarray
        The label for the input image.

    Returns:
    --------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the preprocessed image and label.

    Raises:
    -------
    TypeError
        If the input image, brain mask, or label is not a numpy array.
    ValueError
        If the input image, brain mask, or label is not a 2D array.
        If the input image, brain mask, or label does not have the same shape.

    Notes:
    ------
    This function applies the following preprocessing steps to the input image and mask:
    1. Converts the input image, brain mask, and label to a float32 data type.
    2. Crops the image, brain mask, and label to standard dimensions of 256x256.
    3. Normalizes the image using a Gaussian distribution with zero mean and unit variance over the brain volume.

    Example:
    --------
    >>> image = np.random.rand(176, 256, 256)
    >>> brain_mask = np.random.rand(176, 256, 256)
    >>> label = np.random.rand(176, 256, 256)
    >>> preprocessed_image, preprocessed_label = imagePreProcessing(image, brain_mask, label)
    """
    import numpy as np
    from General_Functions.image_preprocessing import float32_converter, crop_image, gaussian_normalisation

    # Check inputs are numpy arrays and have the correct dimensions
    if not all(isinstance(arr, np.ndarray) for arr in [image, brain_mask, label]):
        raise TypeError("Input image, brain mask, and label must be numpy arrays.")
    if not all(arr.ndim == 2 for arr in [image, brain_mask, label]):
        raise ValueError("Input image, brain mask, and label must be 3D arrays.")
    if not all(arr.shape == image.shape for arr in [brain_mask, label]):
        raise ValueError("Input image, brain mask, and label must have the same shape.")

    # Convert images to float32
    image = float32_converter(image)
    brain_mask = float32_converter(brain_mask)
    label = float32_converter(label)

    # Crop images to standard dimensions (256x256)
    image = crop_image(image)
    brain_mask = crop_image(brain_mask)
    label = crop_image(label)

    # Gaussian normalization over brain volume
    image = gaussian_normalisation(image, brain_mask)

    return image, label