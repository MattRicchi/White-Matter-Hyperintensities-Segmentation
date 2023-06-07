#!/usr/bin/env python3
'''
This script is for plotting flair, ground truth and result images

Author: Mattia Ricchi
Date: May 2023
'''

from setup import Setup_Script

# Check everything is correctly setted before running the script
Setup_Script()

# Import necessary functions and modules
from General_Functions.plot_functions import plot_images

# Read and plot the images
plot_images(38, 147)