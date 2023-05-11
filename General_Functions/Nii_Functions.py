'''
This script contains all the necessary functions to deal with MR images in .nii format.

Author: Mattia Ricchi
Date: May 2023
'''

def readImage(imgPath):
    """
    Function to read and load a .nii file

    Parameters
    ----------
        imgPath (string): Path of the .nii file to be read and loaded . 

    Returns
    -------
        (array): File located at imgPath.
    """
    import nibabel as nib

    return nib.load(imgPath).get_fdata()

def saveSlice(img, fname, path):
    """
    Save an image in the .nii format

    Parameters
    ----------
        img (array): image to be saved.
        fname (string): name with which the image will be saved.
        path (string): where to save the image.
    """
    import os
    import nibabel as nib
    # First check if the path exists and if not make it
    if not os.path.exists(path):
        os.mkdir(path)
    # Then save img as .nii file
    fout = os.path.join(path, f'{fname}.nii')
    nib.save(img, fout)

    return

def sliceAndSaveVolumeImage(vol, fname, path, SLICE_DECIMATE_IDENTIFIER = 3):
    """
    Save a volume image into single slices in the .nii format

    Parameters
    ----------
        vol (array): volume to be saved into slices.
        fname (string): name with which the slices will be saved.
        path (string): where to save the slices.
    """
    import os
    import nibabel as nib
    import numpy as np
    # First check if path exists and if not make it
    if not os.path.exists(path):
        os.mkdir(path)
    # Get volume dimentions
    (number_of_slices, image_rows, image_columns, channel_number) = vol.shape
    # Save volume image into single slices
    for i in range(number_of_slices):
        saveSlice(nib.Nifti1Image(vol[i, :, :, 0], np.eye(4)), f'{fname}', path) 

    return
    
def concatenateImages(flair_img, t1w_img):
    """
    Concatenate two 2D images into one 3D image.

    Parameters
    ----------
        flair_img (array): first image to be concatenated.
        t1w_img (array): second image to be concatenated.
    
    Returns
    -------
        FLAIR_and_T1W_image (array): 3D image containing flair_img and t1w_img.
    """
    import numpy as np
    # Get the shape of the two images
    (image_rows, image_columns) = flair_img.shape
    # Define the channel on which the two images will be concatenated
    channel_number = 2
    # Define the final 3D image
    FLAIR_and_T1W_image = np.ndarray((image_rows, image_columns, channel_number), dtype = np.float32)
    # Create a new axis in the two 2D images
    flair_img = flair_img[..., np.newaxis]
    t1w_img = t1w_img[..., np.newaxis]
    # Finally, concatenate the 2D images into a 3D image
    FLAIR_and_T1W_image = np.concatenate((flair_img, t1w_img), axis = 2)
    return FLAIR_and_T1W_image