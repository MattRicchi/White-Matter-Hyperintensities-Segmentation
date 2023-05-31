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
  * `test_pytest.py` script containing all test functions;
  * `evaluate_results.py` script to compute the Dice Similarity Coefficient, Precision, Recall and F1 score for each test patient and to plot the boxplot of every evaluation metric;
  * `plot_images.py` script to display the flair image, ground turth and segmentation result for every test patient.
* The [General_Functions](https://github.com/MattRicchi/White-Matter-Hyperintensities-Segmentation/tree/main/General_Functions) directory includes three scripts containing the necessary functions to correctly handle with medical images in the NIfTI format, and all the functions for the preprocessing and training stages:
  * `Nii_Functions.py` 
  * `image_preprocessing.py`
  * `Training_Functions.py`
* The `test_folder` contains the images created during the testing of all the functions;
* The [Report](https://github.com/MattRicchi/White-Matter-Hyperintensities-Segmentation/tree/main/Report) directory contains the .tex files used to write the final report of the project.

## Installation

To clone the git repository, type the following commands from terminal:
```
git clone https://github.com/MattRicchi/White-Matter-Hyperintensities-Segmentation.git
cd White-Matter-Hyperintensities-Segmentation
```
The required packages are:
``` 
os
sys
subprocess
```
which do not require to be installed as they are part of the standard Python library. 
The `Setup_Script()` function will install all necessary requirements automatically upon the first launch of the script.

**Always run the `Setup_Script()` function to ensure you have all the packages you need to run the codebase.**

## Running tests

The tests for all the functions, are contained in the `test_pytest.py` script. To run them it is necessary to be in the White-Matter-Hyperintensities-Segmentation directory and to have installed the pytest package. Then it is enough to run the `pytest` command:
```
pytest
```
The scripts were written in `Python 3.11.0` on `Windows 11`, and the functions were tested in the same environment.