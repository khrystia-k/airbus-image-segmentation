# Airbus Image segmentation

This repository contains code for the Airbus Ship Detection Challenge on Kaggle, which involves segmenting ships in satellite images.
For solving this problem were used U-net model architecture and dice score for evaluation.

## Installation
To get started with this project, clone the repository and install the required dependencies:

```
git clone https://github.com/khrystia-k/airbus-image-segmentation.git
cd airbus-image-segmentation
pip install -r requirements.txt
```
Ensure you have the necessary data by downloading it from the Kaggle competition page and placing it in the data/ directory.

## Model usage

To detect airbuses using the model:

1. Download model weights https://drive.google.com/file/d/1Ltpv7t9SETnv7FxCe21rZcGqHqyws-XL/view?usp=sharing
2. Run ```inference.py```

## Example of prediction
![image](https://github.com/user-attachments/assets/d72808d7-3eb7-4615-84be-171e26ca1b3c)

![image](https://github.com/user-attachments/assets/0c012dbb-7999-4056-9ae2-ba89c05a64d8)

![image](https://github.com/user-attachments/assets/db0857f7-36e7-403c-a595-baae9f51a2cf)
