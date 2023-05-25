# White Matter Hyperintensities Segmentation

Within this repository, you will find functions designed to train and test an ensemble model for the purpose of segmenting White Matter Hyperintensities (WMH) in Magnetic Resonance Images. Additionally, this repository includes functions for preprocessing and visualizing MR images.

## Introduction

White matter hyperintensities (WMH) refer to signal abnormalities in white matter on T2-weighted magnetic resonance imaging (MRI) sequences. With the widespread availability of MRI and increasing use of brain magnetic resonance imaging in various clinical settings, clinicians frequently encounter incidental discoveries of white matter lesions, which appear as hyperintensities on T2-weighted images, particularly in FLAIR sequence MRI.

This repository contains an algorithm that can automatically detect and segment white matter hyperintensities by using fluid-attenuated inversion recovery (FLAIR) and T1 magnetic resonance scans. The technique used is based on a deep fully convolutional neural network and ensemble model, which is a machine learning approach for diagnosing diseases through medical imaging.

The repository is structured as follows:

* The main directory includes:
  * `setup.py` script with the `Setup_Script()` function that installs all necessary packages before running any other script;
  * `training.py` script that contains the code to train and test the model;
  * `unet.py` script with the `get_unet()` function defining the network used in the ensemble model;
  * `test_pytest.py` script containing all test functions.
* The [General_Functions](https://github.com/MattRicchi/White-Matter-Hyperintensities-Segmentation/tree/main/General_Functions) directory includes three scripts containing the necessary functions to correctly handle with .nii files, and all the functions for the preprocessing and training stages:
  * `Nii_Functions.py`
  * `image_preprocessing.py`
  * `Training_Functions.py`
  
  