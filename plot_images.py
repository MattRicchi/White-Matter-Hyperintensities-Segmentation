#!/usr/bin/env python3
'''
This script is for plotting 

Author: Mattia Ricchi
Date: May 2023
'''

from setup import Setup_Script

# Check everything is correctly setted before running the script
Setup_Script()

# Import necessary functions and modules
import os
from os.path import join
import matplotlib.pyplot as plt 
from General_Functions.Nii_Functions import readImage
from General_Functions.postprocessing import imagePostProcessing

# Set up necessary paths for data and results
data_path = join(os.getcwd(), 'DATABASE')
results_path = join(os.getcwd(), 'Results')
flair_path = join(data_path, 'flair/')
label_path = join(data_path, 'label/')

# Set the name of the image you want to display
id_ = 'volume-038-147.nii'
volume_number = int(id_[7:10])
slice_number = int(id_[11:14])

# Read flair, label and result images
flair_image = readImage(join(flair_path, id_))
label = readImage(join(label_path, id_))
result = readImage(join(results_path, id_))

# Process the flair and ground truth images
flair_image, label = imagePostProcessing(flair_image, label)

# Define figure size and title
fig = plt.figure(figsize=(15, 6))
plt.title(f'Volume {volume_number} - Slice {slice_number}', fontsize = 25)
plt.axis('off')

rows = 1
columns = 3

# Add flair image at the 1st position
fig.add_subplot(rows, columns, 1)
plt.imshow(flair_image.T, cmap = 'gray')
plt.title("Flair Image", fontsize = 16)

# Add ground truth image at the 2nd position
fig.add_subplot(rows, columns, 2)
plt.imshow(label.T, cmap = 'gray')
plt.title("Label Image", fontsize = 18)

# Add segmentation result at the 3rd position
fig.add_subplot(rows, columns, 3)
plt.imshow(result.T, cmap = 'gray')
plt.title("Segmantation Result", fontsize = 18)