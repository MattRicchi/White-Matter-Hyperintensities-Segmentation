#!/usr/bin/env python3
'''
This script is dedicated to the evaluation of the segmentation results given as output by the algorithm

Author: Mattia Ricchi
Date: June 2023
'''

# Import necessary functions and modules
import os
from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from General_Functions.postprocessing import get_evaluation_metrics
from General_Functions.Nii_Functions import readImage
from General_Functions.image_preprocessing import crop_image

# Set images shape
image_shape = (256, 256, 2)

# Set up necessary paths for ground truth images and results
ground_truth_path = join(os.getcwd(), 'DATABASE/label/')
result_path = join(os.getcwd(), 'Results')

# Set volume numbers used to test the network
labels = []

# Open the file in read mode
with open("test_patients.txt", "r") as file:
    # Read the content of the file
    content = file.read()
    
    # Split the content by spaces
    labels = content.split()

image_ids = next(os.walk(result_path))[2]
patient_number = np.empty(0)

# Initialize dictionaries to store evaluation metrics
dsc = {}
precision = {}
recall = {}
f1 = {}

# Iterate over the predicted images
for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
    # Read and process the predicted and true images
    predicted_image = crop_image(readImage(os.path.join(result_path, f'{id_}')))
    true_image = crop_image(readImage(os.path.join(ground_truth_path, f'{id_}')))
    patient_number = int(id_[7:10])
    
    # Compute evaluation metrics
    dsc_value, precision_value, recall_value, f1_value = get_evaluation_metrics(true_image, predicted_image)
    
    # Store evaluation metrics in the corresponding dictionaries
    if patient_number not in dsc:
        dsc[patient_number] = []
        precision[patient_number] = []
        recall[patient_number] = []
        f1[patient_number] = []
        
    dsc[patient_number].append(dsc_value)
    precision[patient_number].append(precision_value)
    recall[patient_number].append(recall_value)
    f1[patient_number].append(f1_value)

# Create a DataFrame to store the evaluation metrics
df = pd.DataFrame({
    'Patient Number': list(dsc.keys()),
    'DSC': list(dsc.values()),
    'Precision': list(precision.values()),
    'Recall': list(recall.values()),
    'F1-Score': list(f1.values())
    })
df.to_csv('evaluation_metrics.csv', index=False)

# Generate boxplots to visualize the evaluation metrics
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.boxplot(df['DSC'], labels=labels)
plt.title('Dice Similarity Coefficient (DSC)')
plt.xlabel('Patient Number')
plt.ylabel('DSC')

plt.subplot(2, 2, 2)
plt.boxplot(df['Precision'])
plt.title('Precision')
plt.xlabel('Patient Number')
plt.ylabel('Precision')

plt.subplot(2, 2, 3)
plt.boxplot(df['Recall'])
plt.title('Recall')
plt.xlabel('Patient Number')
plt.ylabel('Recall')

plt.subplot(2, 2, 4)
plt.boxplot(df['F1-Score'])
plt.title('F1-Score')
plt.xlabel('Patient Number')
plt.ylabel('F1-Score')

plt.tight_layout()
plt.show()    