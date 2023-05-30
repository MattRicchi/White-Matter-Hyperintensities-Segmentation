#!/usr/bin/env python3
from setup import Setup_Script

Setup_Script()

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from General_Functions.postprocessing import get_evaluation_metrics
from General_Functions.Nii_Functions import readImage
from General_Functions.image_preprocessing import crop_image

image_shape = (256, 256, 2)
smooth = 1
ground_truth_path = 'DATABASE/label/'
result_path = 'Results_LR-5_-6_-6_labelledOnly_NEW/'

labels = ['4', '11', '15', '38', '48', '57']

image_ids = next(os.walk(result_path))[2]
patient_number = np.empty(0)

dsc = {}
precision = {}
recall = {}
f1 = {}

for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
    predicted_image = crop_image(readImage(os.path.join(result_path, f'{id_}')))
    true_image = crop_image(readImage(os.path.join(ground_truth_path, f'{id_}')))
    patient_number = int(id_[7:10])
    
    dsc_value, precision_value, recall_value, f1_value = get_evaluation_metrics(true_image, predicted_image)
    
    if patient_number not in dsc:
        dsc[patient_number] = []
        precision[patient_number] = []
        recall[patient_number] = []
        f1[patient_number] = []
        
    dsc[patient_number].append(dsc_value)
    precision[patient_number].append(precision_value)
    recall[patient_number].append(recall_value)
    f1[patient_number].append(f1_value)


df = pd.DataFrame({
    'Patient Number': list(dsc.keys()),
    'DSC': list(dsc.values()),
    'Precision': list(precision.values()),
    'Recall': list(recall.values()),
    'F1-Score': list(f1.values())
    })
df.to_csv('evaluation_metrics.csv', index=False)

# Generate boxplots
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