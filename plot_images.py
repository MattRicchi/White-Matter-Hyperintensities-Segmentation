#!/usr/bin/env python3
'''
This script is for plotting flair, ground truth and result images

Author: Mattia Ricchi
Date: May 2023
'''

# Import necessary functions and modules
import sys
from General_Functions.plot_functions import plot_images

# Get patient number and slice number to plot
patient_number = sys.argv[1]
slice_number = sys.argv[2]

# Read and plot the images
plot_images(patient_number, slice_number)