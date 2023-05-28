#!/usr/bin/env python3
'''
This script contains all the necessary functions to deal with MR images in .nii format.

Author: Mattia Ricchi
Date: May 2023
'''

def readImage(imgPath):
    """
    Reads in a medical image from a file and returns its data as a NumPy array.

    Parameters
    -----------
    imgPath: str
        The file path of the medical image to be read.

    Returns
    --------
    image: ndarray
        A NumPy array containing the data of the medical image.
    
    Raises
    -------
    FileNotFoundError
        If the specified file path does not exist.
    nibabel.filebasedimages.ImageFileError
        If the specified file path is not a valid medical image file.

    Example
    ---------
    >>> readImage('example.nii.gz')
    array([[[ 0.11,  0.22,  0.33],
            [ 0.44,  0.55,  0.66]],

           [[ 0.77,  0.88,  0.99],
            [ 1.11,  1.22,  1.33]]])
    """
    import nibabel as nib
    
    try:
        # Load the image using nibabel and retrieve its data
        image = nib.load(imgPath).get_fdata()
        return image
    except FileNotFoundError:
        # Raise an error if the file path does not exist
        raise FileNotFoundError(f"File not found: {imgPath}")
    except nib.filebasedimages.ImageFileError:
        # Raise an error if the file path is not a valid medical image file
        raise nib.filebasedimages.ImageFileError(f"Not a valid medical image file: {imgPath}")


def saveSlice(img, fname, path):
    """
    Save an image in the NIfTI format (.nii file extension).

    Parameters
    ----------
    img: array-like
        The image data to be saved. This can be a NumPy array or any other
        array-like object that can be converted to a NumPy array.
    fname: str
        The file name to be used for saving the image, without the file
        extension. For example, if `fname` is "image" then the saved file
        will be named "image.nii".
    path: str
        The directory path where the image file will be saved. If the path
        does not exist, it will be created.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `img` is not an array-like object.
    ValueError
        If `fname` is not a string or if it is an empty string.
        If `path` is not a string or if it is an empty string.
    OSError
        If the file could not be saved for any reason (e.g., the output
        file already exists and cannot be overwritten, or the output
        directory cannot be created).

    Example
    --------
    >>> import numpy as np
    >>> img = np.random.rand(10, 10, 10)
    >>> saveSlice(img, "image", "/path/to/output/dir")
    """
    import os
    import numpy as np
    import nibabel as nib

    # Check inputs for correctness
    if not isinstance(img, np.ndarray):
        raise TypeError("Expected `img` to be an array-like object.")

    if not isinstance(fname, str) or not fname:
        raise ValueError("`fname` must be a non-empty string.")

    if not isinstance(path, str) or not path:
        raise ValueError("`path` must be a non-empty string.")

    # Create output directory if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)

    # Save image to file
    try:
        img = nib.Nifti1Image(img, np.eye(4))
        fout = os.path.join(path, f"{fname}.nii")
        nib.save(img, fout)
    except Exception as e:
        raise OSError(f"Failed to save image to file: {e}")

    return

    
def concatenateImages(flair_img, t1w_img):
    """
    Concatenate two 2D images along a new channel axis to create a 3D image.

    Parameters
    ----------
    flair_img: array-like
        A NumPy array containing the first 2D image to be concatenated.
    t1w_img: array-like
        A NumPy array containing the second 2D image to be concatenated.

    Returns
    -------
    FLAIR_and_T1W_image: ndarray
        A 3D NumPy array containing the two 2D images concatenated along the third dimension.

    Raises
    ------
    TypeError 
        If either flair_img or t1w_img is not a numpy.ndarray.
    ValueError
        If the shape of the two input images is not the same.
        If either flair_img or t1w_img does not have exactly two dimensions.

    Notes
    -----
    The input images must be 2D and have the same shape.

    Example
    --------
    >>> flair_img = np.array([[1, 2], [3, 4]])
    >>> t1w_img = np.array([[5, 6], [7, 8]])
    >>> concatenateImages(flair_img, t1w_img)
    array([[[1., 5.],
            [2., 6.]],

           [[3., 7.],
            [4., 8.]]], dtype=float32)

    """
    import numpy as np

    # Check inputs for correctness
    if not isinstance(flair_img, np.ndarray) or not isinstance(t1w_img, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    
    if flair_img.shape != t1w_img.shape:
        raise ValueError("Input images must have the same shape.")
    
    if flair_img.ndim != 2 or t1w_img.ndim != 2:
        raise ValueError("Inputs must be 2D numpy arrays.")

    # Get the shape of the two images
    (image_rows, image_columns) = flair_img.shape

    # Define the dimention of the channel on which the two images will be concatenated
    channel_number = 2

    # Define the final 3D image
    FLAIR_and_T1W_image = np.ndarray((image_rows, image_columns, channel_number), dtype=np.float32)

    # Create a new axis in the two 2D images
    flair_img = flair_img[..., np.newaxis]
    t1w_img = t1w_img[..., np.newaxis]

    # Finally, concatenate the 2D images into a 3D image
    FLAIR_and_T1W_image = np.concatenate((flair_img, t1w_img), axis=2)

    return FLAIR_and_T1W_image
