import pandas as pd
import numpy as np
import os
from skimage.io import imread
import random

import tensorflow as tf
from utilities import *
from model import *

from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator


BATCH_SIZE = 36
VALIDATION_BATCH = 600
MAX_TRAIN_STEPS = 100
NUM_EPOCHS  = 10
IMG_SCALING =  (3, 3)

folder_path = '/kaggle/input/airbus-ship-detection'
train_path = '/kaggle/input/airbus-ship-detection/train_v2'
test_path = '/kaggle/input/airbus-ship-detection/train_v2'

datagen_args = dict(
    rotation_range=90,  # randomly rotate images by up to 90 degrees
    width_shift_range=0.1,  # randomly shift images horizontally by up to 10%
    height_shift_range=0.1,  # randomly shift images vertically by up to 10%
    shear_range=0.02,  # shear angle in counter-clockwise direction in degrees
    zoom_range=0.2,  # randomly zoom into images by up to 20%
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=True,  # randomly flip images vertically
    fill_mode='reflect')
    
image_gen = ImageDataGenerator(**datagen_args)
mask_gen = ImageDataGenerator(**datagen_args)

# Generators
def batch_gen(batch_size, images, masks):
    
    out_rgb = []
    out_mask = []
    
    while True:
        # random shuffle
        combined = list(zip(images, masks))
        random.shuffle(combined)

        for image, mask in combined:
            out_rgb.append(image)
            out_mask.append(mask)

            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []


def augment_generator(batch_gen, seed = 67):
    np.random.seed(seed)
    for x, y in batch_gen:
        
        batch_size = x.shape[0]
        g_x = image_gen.flow(255*x, batch_size = batch_size, seed = seed)
        g_y = mask_gen.flow(y, batch_size = batch_size, seed = seed)
        
        yield next(g_x)/255.0, next(g_y)


IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 3


input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = build_unet(input_shape)
model.compile(optimizer=Adam(1e-3), loss=dice_loss, metrics=[IoU, dice_coef ])
model.summary()

# Data processing
masks = pd.read_csv(os.path.join(folder_path,'train_ship_segmentations_v2.csv'))
masks['NumOfShips'] = masks['EncodedPixels'].apply(lambda x: 1 if isinstance(x, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'NumOfShips': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['NumOfShips'].map(lambda x: 1 if x > 0 else 0)
unique_img_ids = unique_img_ids[~unique_img_ids['ImageId'].isin(['6384c3e78.jpg'])] # remove corrupted file
masks.drop(['NumOfShips'], axis=1, inplace=True)

# Split to train & validation
train_ids, valid_ids = train_test_split(unique_img_ids, 
                 test_size = 0.3, 
                 stratify = unique_img_ids['NumOfShips'])


train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)

# Balanve dataset
balanced_train_df = balance_df(train_df, 200)
balanced_valid_df = balance_df(valid_df, 50)

print(balanced_train_df.shape[0], 'training masks')
print(balanced_valid_df.shape[0], 'validation masks')

train_x, train_y = df_rle_to_image(balanced_train_df, train_folder_path = train_path)
valid_x, valid_y = df_rle_to_image(balanced_valid_df, train_folder_path = train_path)

valid_x = np.asarray(valid_x, dtype = 'uint8')
valid_y = np.asarray(valid_y, dtype = 'float')

# Train
step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0]//BATCH_SIZE)
aug_gen = augment_generator(batch_gen(BATCH_SIZE, train_x, train_y))
model.fit(aug_gen, 
         steps_per_epoch=step_count, 
         epochs=NUM_EPOCHS, 
         validation_data=(valid_x, valid_y)
                   )

# Save the weights
model.save_weights('/kaggle/working/seg_model_2.weights.h5')

