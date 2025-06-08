"""
Implements the inference pipeline. It accepts a folder of input images and saves the resulting masks in the result/ directory.
"""
import torch
import numpy as np
import cv2
import os
from glob import glob
import albumentations as A
from training import get_model
from image_preprocessing import AlbumentationsTransform,CustomImageEnhancement,fix_shape_fn
from monai.transforms import Transform
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ResizeWithPadOrCropd, EnsureTyped, MapTransform,LambdaD
)
import torch.nn.functional as F
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.spatial import distance_matrix
import networkx as nx

from config import CLASS_LABEL, INFERENCE_MODEL, INFERENCE_IMAGES_PATH, CLASS_SHAPE, INFERENCE_RESULT_PATH,INFERENCE_MASKS_PATH,CLASS_OVERLAP_FIX,SAVE_INFERENCE_MASK

import time
from util import create_folder

albumentations_inference = A.Compose([
        CustomImageEnhancement()
        ])

def get_inference_image_details(folder_path):
    """
    get images from folder.
    """
    extensions=(".png", ".jpg", ".jpeg")
    image_paths=[]
    for ext in extensions:
        image_paths.extend(glob(os.path.join(folder_path, f"*{ext}")))
    image_dicts = [{"image": path} for path in sorted(image_paths)]
    return image_dicts



class ModelSegmenter:
    """
    Class to load the Unet model and do the inference.
    """
    def __init__(self, model_path):

        self.model = get_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load trained checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        # Load only the model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # # Inference-time transform (for grayscale images)
        # self.infer_transform = Compose([
        #     LoadImage(image_only=True),
        #     EnsureChannelFirst(),
        #     ScaleIntensity()
        # ])
       
        self.infer_transform =  self.eval_transform = Compose([
            LoadImaged(keys=["image"]),
            LambdaD(keys=["image"], func=fix_shape_fn),
            EnsureChannelFirstd(keys=["image"]),
            AlbumentationsTransform(
                keys=["image"],
                transform=albumentations_inference
            ),
            ScaleIntensityd(keys=["image"]),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=(480, 640)),
            EnsureTyped(keys=["image"])
        ])

    def predict(self, data):
        data = self.eval_transform(data)
        image_tensor = data["image"].unsqueeze(0).to(self.device)  # [1, C, H, W]

        with torch.no_grad():
            logits = self.model(image_tensor)  # [1, num_classes, H, W]
            #pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)  # [H, W]
            pred = torch.argmax(logits, dim=1).cpu()  # [H, W]

            # Step 1: One-hot encode -> shape [B, H, W, C]
            pred_one_hot = F.one_hot(pred, num_classes=(len(CLASS_LABEL.keys())+1))

            # Step 2: Permute to [B, C, H, W]
            pred_one_hot = pred_one_hot.permute(0, 3, 1, 2).float()

            masks =[]
            for c in range(1,len(CLASS_LABEL.keys())+1):
                masks.append((pred_one_hot[0,c]*255).astype(np.uint8))

            #pred = (pred * int(255/len(CLASS_LABEL.keys()))).astype(np.uint8)

        return masks
    

def fix_mask_gaps(masks):
    """
    Fix the overlap mask gap issue.
    """
    fixed_masks =[]
    for i,mask in enumerate(masks):
        if (i+1) in CLASS_OVERLAP_FIX:
            overlap_class = CLASS_OVERLAP_FIX[i+1]
            overlap_mask = masks[overlap_class-1]

            # Dilate mask  to expand into gaps
            kernel = np.ones((5, 5), np.uint8)
            mask_dilated = cv2.dilate(mask, kernel, iterations=3)

            recovered = cv2.bitwise_and(mask_dilated, overlap_mask)
            recovered = (recovered > 0).astype(np.uint8) * 255

            if np.count_nonzero(recovered) == 0:
                fixed_masks.append(mask)
            else:
                # Combine recovered region with original mask
                fixed_mask = cv2.bitwise_or(mask, recovered)
                fixed_mask = (fixed_mask > 0).astype(np.uint8) * 255
                fixed_masks.append(fixed_mask)
        else:
            fixed_masks.append(mask)
    
    return fixed_masks


def filter_and_merge_line(mask, type, area_threshold=500):
    """
    Post-process MID or EPI masks to get one main line contour.

    Args:
        mask (H,W) binary numpy array
        area_threshold: ignore contours smaller than this

    Returns:
        approximate line
    """
    #mask_uint8 = (mask * 255).astype(np.uint8)

    # Define a square structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Apply Opening (erosion followed by dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply Closing (dilation followed by erosion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]
    if not filtered:
        return None  # no valid contour
    
    max_area_contour = max(filtered, key=cv2.contourArea) # Get maximum area
    
    # Create blank output image with same height and width
    mask = np.zeros(mask.shape, dtype=np.uint8)

    # Draw filled contours
    cv2.drawContours(mask, [max_area_contour], -1, 255, thickness=cv2.FILLED)


    skeleton = skeletonize(mask)
    mask = (skeleton.astype(np.uint8)) * 255

    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=30,minLineLength=5, maxLineGap=30)

    if lines is None:
        return None

    # Find the line with the maximum length
    best_line = max(lines, key=lambda l: np.linalg.norm([l[0][2] - l[0][0], l[0][3] - l[0][1]]))


    # # Merge all contours points into one set of points
    # all_points = np.vstack(filtered)

    # # Option 1: convex hull (encloses all fragments)
    # hull = cv2.convexHull(all_points)

    # # Option 2: approximate polyline (simplify)
    # epsilon = 0.01 * cv2.arcLength(hull, True)
    # approx_line = cv2.approxPolyDP(hull, epsilon, False)

    return tuple(best_line[0])


def get_largest_rectangle(mask, area_threshold=500):
    """
    Post-process VTB mask to get bounding rectangle of largest region.

    Args:
        mask (H,W) binary numpy array
        area_threshold: ignore small contours

    Returns:
        Largest rectangle or None
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Pick largest contour by area
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < area_threshold:
        return None

    bbox = cv2.minAreaRect(largest)  # returns (center (x,y), (width, height), angle of rotation)
    #bbox = cv2.boundingRect(largest)
    box = cv2.boxPoints(bbox)    # Get 4 corners of the rect
    box = np.int0(box)           # Convert to integer
    return box


def post_process_masks(masks):
    """
    Do the post process on predicted masks.
    """
    preprocess_list=[]
    for i,mask in enumerate(masks):
        if CLASS_SHAPE[i+1] == "LINE":
            line = filter_and_merge_line(mask,i)
            preprocess_list.append(["LINE",line])
        elif CLASS_SHAPE[i+1] == "RECTANGLE":
            rect = get_largest_rectangle(mask)
            preprocess_list.append(["RECTANGLE",rect])
        else:
            preprocess_list.append([None,None])

    return preprocess_list

def draw_mask(image_path,preprocessed_list):
    """
    Draw masks on the fimal result image.
    """
    image  = cv2.imread(image_path,0)

    for processed in preprocessed_list:
        if processed[0] == "LINE":
            if processed[1] is not None:
                #image = cv2.polylines(image, [processed[1]], False, 255, 2)
                x1, y1, x2, y2 = processed[1]
                image = cv2.line(image, (x1, y1), (x2, y2), 255, 2)
        elif processed[0] == "RECTANGLE":
            if processed[1] is not None:
                #x,y,w,h = processed[1]
                #image = cv2.rectangle(image, (x,y), (x+w,y+h), 255, 2)  # red rect
                image= cv2.drawContours(image, [processed[1]], 0, 255, 2) 

    save_image_path = INFERENCE_RESULT_PATH + os.path.basename(image_path)
    cv2.imwrite(save_image_path, image)
    print("Inference result saved to : ", save_image_path)

def logical_processing(post_processed_mask_list):
    """
    VTB, EPI will be present only if MID is present.
    If MID not present make others None
    """
    mid_label = CLASS_LABEL["MID"] - 1
    if post_processed_mask_list[mid_label][1] == None:
        for processed in post_processed_mask_list:
            processed[1] = None
    
    return post_processed_mask_list


#TESTING
"""
Below is example on how to run this step.
"""
if __name__ == "__main__":
    create_folder(INFERENCE_RESULT_PATH)
    create_folder(INFERENCE_MASKS_PATH)
    modelSegmenter = ModelSegmenter(INFERENCE_MODEL)
    image_list = get_inference_image_details(INFERENCE_IMAGES_PATH)

    for image in image_list:
        start = time.time()
        print("Processing " , image["image"])
        masks = modelSegmenter.predict(image)
        masks = fix_mask_gaps(masks)

        if SAVE_INFERENCE_MASK:
            for i,mask in enumerate(masks):
                cv2.imwrite(INFERENCE_MASKS_PATH + str(i)+os.path.basename(image["image"]),mask)

        post_processed_mask_list = post_process_masks(masks)
        post_processed_mask_list = logical_processing(post_processed_mask_list)

        end = time.time()
        print(f"Execution time: {end - start:.4f} seconds")

        draw_mask( image["image"], post_processed_mask_list)

        



