#!/usr/bin/env python3

from setup import Setup_Script

# Check everything is correctly setted before running the script
Setup_Script()

from os.path import join
import matplotlib.pyplot as plt 
from General_Functions.Nii_Functions import readImage
from General_Functions.postprocessing import imagePostProcessing

# Define all necessary paths 
data_path = 'path/to/DATABASE/'
results_path = 'path/to/Results'
flair_path = join(data_path, 'flair/')
label_path = join(data_path, 'label/')

# Set the name of the image you want to display
id_ = 'volume-038-147.nii'

# Read flair, label and result images
flair_image = readImage(join(flair_path, id_))
label = readImage(join(label_path, id_))
result = readImage(join(results_path, id_))

flair_image, label = imagePostProcessing(flair_image, label)

fig = plt.figure(figsize=(15, 6))
plt.title(f'Volume {int(id_[7:10])} - Slice {int(id_[11:14])}', fontsize = 25)
plt.axis('off')

rows = 1
columns = 3

# Add a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
plt.imshow(flair_image.T, cmap = 'gray')
plt.title("Flair Image", fontsize = 16)

# Add a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
plt.imshow(label.T, cmap = 'gray')
plt.title("Label Image", fontsize = 18)

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
plt.imshow(result.T, cmap = 'gray')
plt.title("Segmantation Result", fontsize = 18)