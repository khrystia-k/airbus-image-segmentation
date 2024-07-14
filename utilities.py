
import numpy as np
import pandas as pd
from skimage.io import imread


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T # added tranpose here


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)

    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def balance_df(df, min_count = None):
    
    if not min_count:
        min_count = df['NumOfShips'].value_counts().min()
        
    balanced_df = df.groupby('NumOfShips').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    return df[df['ImageId'].isin(balanced_df['ImageId'])]


def df_rle_to_image(df, train_folder_path, IMG_SCALING=(3, 3) ):
    img_ids = list(df.groupby('ImageId'))
    
    out_rgbs = []
    out_masks = []
    
    for img_info in img_ids:
        img_id = img_info[0]
        img_rgb = imread(os.path.join(train_folder_path, img_id))
        img_mask = masks_as_image(df.loc[df['ImageId'] == img_id, 'EncodedPixels'].tolist())
     
        if IMG_SCALING is not None:
                img_rgb = img_rgb[::IMG_SCALING[0], ::IMG_SCALING[1]]
                img_mask = img_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]

        out_rgbs += [img_rgb]
        out_masks += [img_mask]

    return out_rgbs, out_masks