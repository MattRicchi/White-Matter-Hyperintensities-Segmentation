#!/usr/bin/env python3
'''
This script contains the function get_unet that defines the Network used in the ensamble model

Author: Mattia Ricchi
Date: May 2023
'''

def get_unet(inputs):

    import os
    import tensorflow.keras as keras
    from tensorflow.keras import layers as L
    from General_Functions.Training_Functions import get_crop_shape

    conv1 = L.Conv2D(filters=64, kernel_size=5, strides=1, activation='relu',
                     padding='same', data_format='channels_last', name='conv1_1')(inputs)
    conv1 = L.Conv2D(filters=64, kernel_size=5, strides=1, activation='relu',
                     padding='same', data_format='channels_last', name='conv1_2')(conv1)
    pool1 = L.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool1')(conv1)

    conv2 = L.Conv2D(filters=96, kernel_size=3, strides=1, activation='relu', padding='same',  data_format='channels_last', name='conv2_1')(pool1)
    conv2 = L.Conv2D(96, 3, 1, activation='relu', padding='same',  data_format='channels_last', name='conv2_2')(conv2)
    pool2 = L.MaxPooling2D(pool_size=(2, 2),  data_format='channels_last', name='pool2')(conv2)

    conv3 = L.Convolution2D(128, 3, 1, activation='relu', padding='same', data_format='channels_last', name='conv3_1')(pool2)
    conv3 = L.Convolution2D(128, 3, 1, activation='relu', padding='same', data_format='channels_last', name='conv3_2')(conv3)
    pool3 = L.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool3')(conv3)

    conv4 = L.Conv2D(256, 3, 1, activation='relu', padding='same', data_format='channels_last', name='conv4_1')(pool3)
    conv4 = L.Conv2D(256, 4, 1, activation='relu', padding='same', data_format='channels_last', name='conv4_2')(conv4)
    pool4 = L.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='pool4')(conv4)

    conv5 = L.Conv2D(512, 3, 1, activation='relu', padding='same', data_format='channels_last', name='conv5_1')(pool4)
    conv5 = L.Conv2D(512, 3, 1, activation='relu', padding='same', data_format='channels_last', name='conv5_2')(conv5)

    up_conv5 = L.UpSampling2D(size=(2, 2), data_format='channels_last', name='up_sampling1')(conv5)
    
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = L.Cropping2D(cropping=(ch,cw), data_format='channels_last', name='crop_conv1')(conv4)
    up6 = L.Concatenate(axis=-1, name='concatenate_1')([up_conv5, crop_conv4])
    conv6 = L.Conv2D(256, 3, 1, activation='relu', padding='same', data_format='channels_last', name='conv6_1')(up6)
    conv6 = L.Conv2D(256, 3, 1, activation='relu', padding='same', data_format='channels_last', name='conv6_2')(conv6)

    up_conv6 = L.UpSampling2D(size=(2, 2), data_format='channels_last', name='up_sampling2')(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = L.Cropping2D(cropping=(ch,cw), data_format='channels_last', name='crop_conv2')(conv3)
    up7 = L.Concatenate(axis=-1, name='concatenate_2')([up_conv6, crop_conv3])
    conv7 = L.Conv2D(128, 3, 1, activation='relu', padding='same', data_format='channels_last')(up7)
    conv7 = L.Conv2D(128, 3, 1, activation='relu', padding='same', data_format='channels_last')(conv7)

    up_conv7 = L.UpSampling2D(size=(2, 2), data_format='channels_last', name='up_sampling3')(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = L.Cropping2D(cropping=(ch,cw), data_format='channels_last', name='crop_conv3')(conv2)
    up8 = L.Concatenate(axis=-1, name='concatenate_3')([up_conv7, crop_conv2])
    conv8 = L.Conv2D(96, 3, 1, activation='relu', padding='same', data_format='channels_last')(up8)
    conv8 = L.Conv2D(96, 3, 1, activation='relu', padding='same', data_format='channels_last')(conv8)

    up_conv8 = L.UpSampling2D(size=(2, 2), data_format='channels_last', name='up_sampling4')(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = L.Cropping2D(cropping=(ch,cw), data_format='channels_last', name='crop_conv4')(conv1)
    up9 = L.Concatenate(axis=-1, name='concatenate_4')([up_conv8, crop_conv1])
    conv9 = L.Conv2D(64, 3, 1, activation='relu', padding='same', data_format='channels_last')(up9)
    conv9 = L.Conv2D(64, 3, 1, activation='relu', padding='same', data_format='channels_last')(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = L.ZeroPadding2D(padding=(ch, cw), data_format='channels_last')(conv9)
    conv10 = L.Conv2D(1, 1, 1, activation='sigmoid', data_format='channels_last')(conv9)

    model = keras.Model(inputs, conv10)
    
    return model