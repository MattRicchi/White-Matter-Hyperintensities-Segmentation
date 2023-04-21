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
    