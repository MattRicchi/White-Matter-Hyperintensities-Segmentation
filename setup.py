#!/usr/bin/env python3
'''
This script contains the Setup_Script function that installs all the required packages.
Please, always run this to ensure you have all the required packages installed.

Author: Mattia Ricchi
Date: May 2023
'''

def Setup_Script():
    '''
    This function checks for the availability of necessary packages (numpy, nibabel, tqdm, 
    tensorflow, focal_loss and platform) in the current environment. 
    If the packages are not installed, it installs them using pip. 
    Once all the packages are installed, the function prints the CPU platform and a message 
    to indicate the setup process is complete.

    Parameters
    ----------
    None. 

    Returns
    -------
    None.

    Example
    -------
    >>> Setup_Script()
        /path/to/current/directory
        Checking you have all packages needed for the codebase...
        Using cpu: Intel(R) Core(TM) i5-1035G1 CPU @ 1.00GHz
        All good, off we go!
    '''

    import sys
    import subprocess
    import os

    sys.path.append(os.getcwd())
    print(os.getcwd())

    print('Checking you have all packages needed for the codebase...')

    try:
        import numpy
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
    try:
        import nibabel
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nibabel'])
    try:
        import tqdm
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
    try:
        import tensorflow
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    try:
        import focal_loss
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'focal-loss'])
    try:
        import platform
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'platform'])
    try:
        import cv2
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python'])
    
    print('Using cpu: ' + platform.processor())
    print('All good, off we go!')

    return