# Airbus Image segmentation

This repository contains code for the [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection/overview) on Kaggle, which involves segmenting ships in satellite images.

## Solution

First, EDA was performed and the dataset was found to be very unbalanced. To solve this problem, the data was balanced (a dataset was formed with the same number of images without ships, with one ship, and so on).
Next, all masks were decoded from  Run Length Encoding  (RLE).
The images and masks were resized to a standard size of 256x256 pixels for ease of processing. 

For solving this problem U-net model architecture, with (classic, with the addition of Batch Normalization) was used and dice score for evaluation.

## Model usage

To detect airbuses using the model:

1. Download model weights https://drive.google.com/file/d/1Ltpv7t9SETnv7FxCe21rZcGqHqyws-XL/view?usp=sharing
2. Run ```model-inference.py```

## Example of prediction
![image](https://github.com/user-attachments/assets/d72808d7-3eb7-4615-84be-171e26ca1b3c)

![image](https://github.com/user-attachments/assets/0c012dbb-7999-4056-9ae2-ba89c05a64d8)

![image](https://github.com/user-attachments/assets/db0857f7-36e7-403c-a595-baae9f51a2cf)

## Files
* ```airbus-eda.ipynb``` - Explanatory Data Analysis 
* ``` utilities.py``` - utility functions
* ```model.py ``` - model architecture
* ``` model-training.py``` - model training
* ``` model-inference.py ``` - testing model


