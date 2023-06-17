#!/usr/bin/env python3
'''
This script is dedicated to the training and testing of the model

Author: Mattia Ricchi
Date: May 2023
'''

# Import necessary packages and modules
import os
import time
import nibabel as nib
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras
from os.path import join
from tensorflow.keras import layers as L
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from focal_loss import BinaryFocalLoss, binary_focal_loss

# Import necessary functions and modules
from unet import get_unet
from General_Functions.Training_Functions import dice_coef_loss, scheduler,  get_test_patients, build_train_test_data
from General_Functions.Nii_Functions import saveSlice

Total_Start = time.time()

# Set up necessary paths for data, results, and weights
data_path = join(os.getcwd(), 'DATABASE')
results_path = join(os.getcwd(), 'Results') # Folder where all result images will be saved
flair_path = join(data_path, 'OnlyBrain/flair/')
weights_path = join(os.getcwd(), 'weights/') # Folder where final weights of the network will be saved

# Define the id of test patients
test_patients_file = 'test_patients.txt'
test_patients = get_test_patients(test_patients_file)

# Define the shape of the input images
image_shape = (256, 256, 2)

# This will be passed as an input to the three models of the ensemble
inputs = L.Input(shape=image_shape)

print('Loading and compiling the three models of the ensemble.')

# First model of the ensemble
model0 = get_unet(inputs)
_ = model0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=dice_coef_loss, metrics=[dice_coef_loss])

# Second model of the ensemble
model1 = get_unet(inputs)
_ = model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss=BinaryFocalLoss(gamma=1), metrics=[binary_focal_loss])

# Third model of the ensemble
model2 = get_unet(inputs)
_ = model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=BinaryFocalLoss(gamma=2), metrics=[binary_focal_loss])

# Print model summaries for an overview of the architectures to check that they are correctly compiled
model0.summary()
model1.summary()
model2.summary()

# Define model Checkpoint and Callbacks
checkpointer = ModelCheckpoint('model_for_hyperintensities.h5', verbose = 1, save_weights_only = True)
callback = [keras.callbacks.LearningRateScheduler(scheduler, verbose=1), keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15)]

# Get the ID of the images 
image_ids = next(os.walk(flair_path))[2]

# Get the ID of images containing lesions
labeled_ids = open("DATABASE/labeled_slices.txt", "r")
labeled_ids = labeled_ids.read()

# Define the arrays for the Train and Test images
TRAIN_IMAGES = np.ndarray((0, image_shape[0], image_shape[1], image_shape[2]), dtype = np.float32)
TEST_IMAGES = np.ndarray((0, image_shape[0], image_shape[1], image_shape[2]), dtype = np.float32)
TRAIN_LABELS = np.zeros((0, image_shape[0], image_shape[1]), dtype = np.float32)
Image_IDs = np.empty(0)

# Iterate over the image IDs and label the images as Train or Test image
print('Building X_train, Y_train and X_test... ')
for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
    TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, Image_IDs = build_train_test_data(data_path, test_patients, labeled_ids, id_, 
                                                                               TEST_IMAGES, TRAIN_IMAGES, TRAIN_LABELS, Image_IDs)

# Correct the shape of TRAIN_LABELS
TRAIN_LABELS = TRAIN_LABELS[..., np.newaxis]

# Fit the models
print('Starting to fit the models in the enseble, will take a while...')

start = time.time()

_ = model0.fit(TRAIN_IMAGES, TRAIN_LABELS, validation_split = 0.1, batch_size = 30, epochs = 50, verbose = 1, callbacks = callback)
_ = model1.fit(TRAIN_IMAGES, TRAIN_LABELS, validation_split = 0.1, batch_size = 30, epochs = 50, verbose = 1, callbacks = callback)
_ = model2.fit(TRAIN_IMAGES, TRAIN_LABELS, validation_split = 0.1, batch_size = 30, epochs = 50, verbose = 1, callbacks = callback)

end = time.time()

# Print a message indicating the completion of the training
print('Training completed successfully!')
print('Time spent training the network: ',(end - start)/(60*60), ' hours')

print('Starting to test the network...')

start = time.time()

# Test the three models of the ensemble
preds_test_0 = model0.predict(TEST_IMAGES, verbose = 1)
preds_test_1 = model1.predict(TEST_IMAGES, verbose = 1)
preds_test_2 = model2.predict(TEST_IMAGES, verbose = 1)

# Average the results from the three models of the ensamble
preds_test = (preds_test_0 + preds_test_1 + preds_test_2)/3

# Threshold the result
preds_test_t = (preds_test > 0.4).astype(np.uint8)

end = time.time()

print('Time spent for testing the network: ',(end - start)/((60)), ' minutes')

# Save final results in the form of NIfTI images
print('Saving results image volume... ')
for i in tqdm(range(preds_test_t.shape[0])):
    SLICE_DECIMATE_IDENTIFIER = 3
    patient_name = Image_IDs[i]
    patient_number = int(patient_name[7:10])
    slice_number = int(patient_name[11:14])
    saveSlice(nib.Nifti1Image(preds_test_t[i, :, :, 0], np.eye(4)), f'volume-{str(patient_number).zfill(SLICE_DECIMATE_IDENTIFIER)}-{str(slice_number).zfill(SLICE_DECIMATE_IDENTIFIER)}', results_path)
    
# Save the final weights of the models
print('Saving the final weights of the models...')
model0.save_weights(join(weights_path, 'model0.h5'))
model1.save_weights(join(weights_path, 'model1.h5'))
model2.save_weights(join(weights_path, 'model2.h5'))

Total_End = time.time()

print("All done! Total time for script: ", (Total_End - Total_Start)/(60*60), ' hours')