from model import *
from utilities import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib as plt

model = build_unet((256, 256, 3))
model.load_weights('/kaggle/input/models-weights/seg_model.weights.h5')

folder_path = '/kaggle/input/airbus-ship-detection'
train_path = '/kaggle/input/airbus-ship-detection/train_v2'

# Data processing
masks = pd.read_csv(os.path.join(folder_path,'train_ship_segmentations_v2.csv'))
masks['NumOfShips'] = masks['EncodedPixels'].apply(lambda x: 1 if isinstance(x, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'NumOfShips': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['NumOfShips'].map(lambda x: 1 if x > 0 else 0)
unique_img_ids = unique_img_ids[~unique_img_ids['ImageId'].isin(['6384c3e78.jpg'])] # remove corrupted file
masks.drop(['NumOfShips'], axis=1, inplace=True)


train_ids, valid_ids = train_test_split(unique_img_ids, 
                 test_size = 0.3, 
                 stratify = unique_img_ids['NumOfShips'])

valid_df = pd.merge(masks, valid_ids)
balanced_valid_df = balance_df(valid_df, 50)
valid_x, valid_y = df_rle_to_image(balanced_valid_df, train_folder_path = train_path)

valid_x = np.asarray(valid_x, dtype = 'uint8')

# Predict masks
pred_y = model.predict(valid_x)

# Plot the predictions
num_images = 8

fig, axarr = plt.subplots(num_images, 3, figsize=(15, 4 * num_images))

for i in range(num_images):
    axarr[i, 0].imshow(valid_x[i ])
    axarr[i, 1].imshow(valid_y[i ])
    axarr[i, 2].imshow(pred_y[i ])
    
    axarr[i, 0].axis('off')
    axarr[i, 1].axis('off')
    axarr[i, 2].axis('off')

    axarr[i, 0].set_title('Original Image')
    axarr[i, 1].set_title('True Mask')
    axarr[i, 2].set_title('Predicted Mask')

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()