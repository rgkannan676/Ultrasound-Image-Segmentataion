"""
Handles image preprocessing and augmentation for training and testing. 
The processed images and masks are saved under the ./Train_Test_Dataset/ directory.
"""
import os
import cv2
import albumentations as A
from PIL import Image
from copy import deepcopy
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ResizeWithPadOrCropd, EnsureTyped, MapTransform,LambdaD,AsDiscreted
)
from config import IMAGE_FOLDER,TEMP_MASK_FOLDER,NUMBER_OF_AUGMENTATAIONS,TRAIN_SPLIT_CSV,TEST_SPLIT_CSV,CLASS_LABEL,TRAIN_TEST_FOLDER,SAVE_TRAIN_TEST_IMAGES
import numpy as np
from monai.data import Dataset
import pandas as pd
import torch

def get_train_test_image_details():
    # Read CSV with  headers
    train_df = pd.read_csv(TRAIN_SPLIT_CSV, header=0)
    test_df = pd.read_csv(TEST_SPLIT_CSV, header=0)

    # Iterate through the first two columns
    train_list = [] 
    for i, row in train_df.iterrows():
        train_list.append({"image": row[0], "label":row[1]})
    
    test_list = [] 
    for i, row in test_df.iterrows():
        test_list.append({"image": row[0], "label":row[1]})

    return train_list,test_list

def noise_reduction(image):
    """
    This function reduces the noise in the ultrasound images
    Using Non-Local Means Denoising as it retains structure better than basic filters. Default recomended values are used currently
    """
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=5, templateWindowSize=7, searchWindowSize=21)
    return denoised_image

def contrast_enhancement(image):
    """
    This function improves contrast by using:
    CLAHE (Contrast Limited Adaptive Histogram Equalization)to boosts contrast locally without over-saturating.  Default recomended values are used currently
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def edge_enhancement(image):
    """
    This function improves edges to makes boundaries more defined and separable
    Edge enhancement using cv2.addWeighted involves sharpening an image by blending it with a blurred version of itself.
    """
    edge_enhanced_image = cv2.addWeighted(image, 1.5, cv2.GaussianBlur(image, (0,0), 3), -0.5, 0)
    return edge_enhanced_image


class CustomImageEnhancement(A.ImageOnlyTransform):
    """
    A custom classfunction extended using albumentations. This will use the methods to enhance ultrasound image quality
    """
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        
    def apply(self, image, **params):
        image = noise_reduction(image)
        image = contrast_enhancement(image)
        image = edge_enhancement(image)
        return image


class AlbumentationsTransform(MapTransform):
    def __init__(self, keys, transform, additional_targets=None):
        super().__init__(keys)
        self.transform = transform
        self.additional_targets = additional_targets or {}

    def __call__(self, data):
        d = dict(data)
        input_data = {k: d[k] for k in self.keys}

        # Handle additional targets (e.g. masks)
        if self.additional_targets:
            for k in self.additional_targets:
                input_data[k] = d[k]

        # Convert to numpy (if tensor)
        for k in input_data:
            if hasattr(input_data[k], "numpy"):
                input_data[k] = input_data[k].detach().cpu().numpy()
            if input_data[k].ndim == 3:
                input_data[k] = input_data[k][0]  # remove channel for Albumentations

        result = self.transform(**input_data)
    
        # Put back into dict
        for k in self.keys:
            d[k] = np.expand_dims(result[k], axis=0).astype(np.float32 if k == "image" else np.uint8)
        return d


def fix_shape_fn(x):
    # Swap axes 0 and 1 if first dim is bigger than second (likely W, H instead of H, W)
    if x.ndim == 3:
        # For image: could be (W, H, C) or (H, W, C)
        if x.shape[0] > x.shape[1]:
            return x.swapaxes(0, 1)
    elif x.ndim == 2:
        # For mask: could be (W, H)
        if x.shape[0] > x.shape[1]:
            return x.swapaxes(0, 1)
    return x   

def create_image_transforms():
    # ======= Albumentations for Training =======
    albumentations_train = A.Compose([
        CustomImageEnhancement(),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.1,0.1), rotate_limit=(-10,10), p=0.4),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=5, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=(-0.1,0.1), p=0.3),
        #A.Downscale(scale_min=0.9, scale_max=0.9, p=0.2), # Issue with grayscale
        #A.GaussianBlur(blur_limit=3, p=0.2) # Issue with grayscale
    ],additional_targets={'label': 'mask'})

    # ======= Albumentations for Validation : Only the image enhancement augmentataion is present. =======
    albumentations_test = A.Compose([
        CustomImageEnhancement()
    ],additional_targets={'label': 'mask'})

    # ======= MONAI Transform Pipelines =======
    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        LambdaD(keys=["image", "label"], func=fix_shape_fn),
        EnsureChannelFirstd(keys=["image", "label"]),
        AlbumentationsTransform(
            keys=["image", "label"],
            transform=albumentations_train
        ),
        ScaleIntensityd(keys=["image"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(480, 640)),
        EnsureTyped(keys=["image", "label"]),
        AsDiscreted(keys=["label"], to_onehot=None, dtype=torch.int64)
    ])

    test_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        LambdaD(keys=["image", "label"], func=fix_shape_fn),
        EnsureChannelFirstd(keys=["image", "label"]),
        AlbumentationsTransform(
                keys=["image"],
                transform=albumentations_test  # Albumentations sees this as a mask
        ),
        ScaleIntensityd(keys=["image"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(480, 640)),
        EnsureTyped(keys=["image", "label"]),
        AsDiscreted(keys=["label"], to_onehot=None, dtype=torch.int64)
    ])

    return train_transform, test_transform


def creat_dataset_for_training():
    """
    This function transforms the train and validation dataset images and masks with augmentataions and changes required for MONAI Unet model
    """
    train_list,test_list = get_train_test_image_details()
    train_transform, test_transform = create_image_transforms()

    #FOR TESTING
    # sample = [{
    #     "image": IMAGE_FOLDER +"Image_97.png",  # grayscale ultrasound image
    #     "label": TEMP_MASK_FOLDER + "Image_97.png"    # 3-class mask
    # }]
    # train_list = sample
    # test_list = sample


    # ======= Output Directories =======
    train_dir = os.path.join(TRAIN_TEST_FOLDER, "train")
    val_dir = os.path.join(TRAIN_TEST_FOLDER, "test")
    os.makedirs(f"{train_dir}/images", exist_ok=True)
    os.makedirs(f"{train_dir}/masks", exist_ok=True)
    os.makedirs(f"{val_dir}/images", exist_ok=True)
    os.makedirs(f"{val_dir}/masks", exist_ok=True)

    # ======= Save Training Samples (Multiple) =======
    train_data_transformed = []
    for i in range(NUMBER_OF_AUGMENTATAIONS):
        train_data_transformed = train_data_transformed + train_transform(deepcopy(train_list))
    
    if SAVE_TRAIN_TEST_IMAGES:
        for i, transformed_train_data in enumerate(train_data_transformed):
            img_np = transformed_train_data["image"][0].cpu().numpy()
            mask_np = transformed_train_data["label"][0].cpu().numpy().astype(np.uint8)

            img_vis = ((img_np - img_np.min()) / (img_np.ptp() + 1e-5) * 255).astype(np.uint8)
            mask_vis = (mask_np * int(255/len(CLASS_LABEL.keys()))).astype(np.uint8)

            Image.fromarray(img_vis).save(f"{train_dir}/images/image_{i}.png")
            Image.fromarray(mask_vis).save(f"{train_dir}/masks/image_{i}.png")
        
        print("Saved Train Images and Mask")

    # ======= Save Validation Sample (Once) =======
    test_data_transformed = test_transform(deepcopy(test_list))
    
    if SAVE_TRAIN_TEST_IMAGES:
        for i,transformed_test_data in enumerate(test_data_transformed):
            img_np = transformed_test_data["image"][0].cpu().numpy()
            mask_np = transformed_test_data["label"][0].cpu().numpy().astype(np.uint8)

            img_vis = ((img_np - img_np.min()) / (img_np.ptp() + 1e-5) * 255).astype(np.uint8)
            mask_vis = (mask_np * 85).astype(np.uint8)

            Image.fromarray(img_vis).save(f"{val_dir}/images/image_test_{i}.png")
            Image.fromarray(mask_vis).save(f"{val_dir}/masks/image_test_{i}.png")

        print("Saved Test Images and Mask")

        print(f"Saved {len(train_data_transformed)} training samples and {len(test_data_transformed)} test sample to: {TRAIN_TEST_FOLDER}")

    return Dataset(data=train_data_transformed),Dataset(data=test_data_transformed)




#TESTING 
"""
Below is example on how to run this step.
"""
if __name__ == "__main__":
    creat_dataset_for_training()

