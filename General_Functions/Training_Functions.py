def dataAugmentation(flair, t1, label):
    import numpy as np
    import tensorflow.keras as keras 
    from keras.preprocessing.image import ImageDataGenerator

    im_gen = ImageDataGenerator()

    theta = np.random.uniform(-15, 15)
    shear = np.random.uniform(-.1, .1)
    zx, zy = np.random.uniform(.9, 1.1, 2)
    flairAug = im_gen.apply_transform(x = flair[..., np.newaxis], transform_parameters={'theta': theta, 'shear': shear, 'zx': zx, 'zy': zy})
    t1Aug = im_gen.apply_transform(x = t1[..., np.newaxis], transform_parameters={'theta': theta, 'shear': shear, 'zx': zx, 'zy': zy})
    labelAug = im_gen.apply_transform(x = label[..., np.newaxis], transform_parameters={'theta': theta, 'shear': shear, 'zx': zx, 'zy': zy})
    return flairAug[:, :, 0], t1Aug[:, :, 0], labelAug[:, :, 0]

def dice_coef_for_training(y_true, y_pred):
    import numpy as np
    import tensorflow.keras as keras
    from keras import backend as K

    smooth = 1
    print(np.shape(y_pred))
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    import numpy as np

    print(np.shape(y_pred))
    print(np.shape(y_true))
    return -dice_coef_for_training(y_true, y_pred)

def get_crop_shape(target, refer):
    '''
    '''
    cw = target.get_shape()[2] - refer.get_shape()[2]
    
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = target.get_shape()[1] - refer.get_shape()[1]

    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def scheduler(epoch, lr):
  import tensorflow as tf
  
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)