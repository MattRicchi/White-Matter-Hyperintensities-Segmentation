#!/usr/bin/env python3

from setup import Setup_Script

# Check everything is correctly setted before running the script
Setup_Script()

# Import necessary packages
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

# Import necessary functions
from unet import get_unet
from General_Functions.Training_Functions import dice_coef_loss, scheduler, dataAugmentation
from General_Functions.image_preprocessing import imagePreProcessing
from General_Functions.Nii_Functions import readImage, concatenateImages, saveSlice

Total_Start = time.time()

# Define all the necessary paths
data_path = join(os.getcwd(), 'DATABASE')
results_path = join(os.getcwd(), 'Results')
flair_path = join(data_path, 'OnlyBrain/flair/')
t1w_path = join(data_path, 'OnlyBrain/t1w/')
label_path = join(data_path, 'OnlyBrain/label/')
brain_path = join(data_path, 'brain/')
weights_path = join(os.getcwd(), 'weights', '{}')

# Define the id of test patients
test_patients = [4, 11, 15, 38, 48, 57]

# Define the shape of the input images
image_shape = (256, 256, 2)

# This will be passed as an input to the three models of the ensemble
inputs = L.Input(shape=image_shape)

print('Loading and compiling the three models of the ensemble')

# First model of the ensemble
model0 = get_unet(inputs)
_ = model0.load_weights(weights_path.format('0.h5'))
_ = model0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=dice_coef_loss, metrics=[dice_coef_loss])

# Second model of the ensemble
model1 = get_unet(inputs)
_ = model1.load_weights(weights_path.format('1.h5'))
_ = model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss=BinaryFocalLoss(gamma=1), metrics=[binary_focal_loss])

# Third model of the ensemble
model2 = get_unet(inputs)
_ = model2.load_weights(weights_path.format('2.h5'))
_ = model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=BinaryFocalLoss(gamma=2), metrics=[binary_focal_loss])

# Visually check that the model is correctly compiled
model0.summary()
model1.summary()
model2.summary()

# Define model Checkpoint and Callbacks
checkpointer = ModelCheckpoint('model_for_hyperintensities.h5', verbose = 1, save_weights_only = True)
callback = [keras.callbacks.LearningRateScheduler(scheduler, verbose=1), keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15)]

# Get the ID of the images 
image_ids = next(os.walk(flair_path))[2]

# Get the ID of labeled images
labeled_ids = open("DATABASE/labeled_slices.txt", "r")
labeled_ids = labeled_ids.read()

# Define the arrays for the Train and Test images
TRAIN_IMAGES = np.ndarray((0, image_shape[0], image_shape[1], image_shape[2]), dtype = np.float32)
TEST_IMAGES = np.ndarray((0, image_shape[0], image_shape[1], image_shape[2]), dtype = np.float32)
TRAIN_LABELS = np.zeros((0, image_shape[0], image_shape[1], 1), dtype = np.float32)
Image_IDs = np.empty(0)

# Iterate over the image IDs and save the images as Train or Test image
print('Building X_train, Y_train and X_test... ')
for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
    flair_image = readImage(join(flair_path, f'{id_}'))
    t1w_image = readImage(join(t1w_path, f'{id_}'))
    label_image = readImage(join(label_path, f'{id_}'))
    brain_mask = readImage(join(brain_path, f'{id_}'))

    (flair_image, label_image) = imagePreProcessing(flair_image, brain_mask, label_image)
    (t1w_image, label_image) = imagePreProcessing(t1w_image, brain_mask, label_image)
    
    if np.max(brain_mask) == 0.0:
        continue # Skip the images in which there is no brain

    FLAIR_and_T1W_image = concatenateImages(flair_image, t1w_image)
    FLAIR_and_T1W_image = FLAIR_and_T1W_image[np.newaxis, ...]
    label_image = label_image[np.newaxis, ..., np.newaxis]
    
    # Sort images out basing on the patient number
    patient_number = int(id_[7:10])
    if patient_number in test_patients:
        # Image will be used for testing
        TEST_IMAGES = np.append(TEST_IMAGES, FLAIR_and_T1W_image, axis = 0)
        Image_IDs = np.append(Image_IDs, image_ids[n])
        
    else:    
        # Image is classified as a training image
        TRAIN_IMAGES = np.append(TRAIN_IMAGES, FLAIR_and_T1W_image, axis = 0)
        TRAIN_LABELS = np.append(TRAIN_LABELS, label_image, axis = 0)
        
        # If there are labelled lesions in the slice, apply data augmentation 10 times
        if id_[:14] in labeled_ids:
            labelImg = label_image[0, :, :, 0]
            for k in range(0, 9):
                flairAug, t1Aug, labelAug = dataAugmentation(flair_image, t1w_image, labelImg)
                FLAIR_and_T1W_image = concatenateImages(flairAug, t1Aug)
                FLAIR_and_T1W_image = FLAIR_and_T1W_image[np.newaxis, ...]
                labelAug = labelAug[np.newaxis, ..., np.newaxis]
                TRAIN_IMAGES = np.append(TRAIN_IMAGES, FLAIR_and_T1W_image, axis = 0)
                TRAIN_LABELS = np.append(TRAIN_LABELS, labelAug, axis = 0)
                k += 1

# Fit the model
print('Starting to fit the models in the enseble, will take a while...')

start = time.time()

_ = model0.fit(TRAIN_IMAGES, TRAIN_LABELS, validation_split = 0.1, batch_size = 30, epochs = 50, verbose = 1, callbacks = callback)
_ = model1.fit(TRAIN_IMAGES, TRAIN_LABELS, validation_split = 0.1, batch_size = 30, epochs = 50, verbose = 1, callbacks = callback)
_ = model2.fit(TRAIN_IMAGES, TRAIN_LABELS, validation_split = 0.1, batch_size = 30, epochs = 50, verbose = 1, callbacks = callback)

end = time.time()

print('Time spent training the network: ',(end - start)/(60*60), ' hours')

print('Starting to test the network...')

start = time.time()

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
    
# Save the weights of the trained network
model0.save_weights(os.path.join(os.getcwd(), 'weights/model0.h5'))
model1.save_weights(os.path.join(os.getcwd(), 'weights/model1.h5'))
model2.save_weights(os.path.join(os.getcwd(), 'weights/model2.h5'))

Total_End = time.time()

print("All done! Total time for script: ", (Total_End - Total_Start)/(60*60), ' hours')