"""
Implements logic for splitting the dataset into training and testing sets. 
It also parses annotation JSON files and generates corresponding masks. 
The resulting splits are saved as train_split.csv and test_split.csv.
"""
import cv2
import numpy as np
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm
from util import create_folder
from config import IMAGE_FOLDER,ANNOTATAION_PATH,CLASS_LABEL,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_TYPE,TRAIN_TEST_SPLIT_RATIO,TEMP_MASK_FOLDER,TRAIN_SPLIT_CSV,TEST_SPLIT_CSV

def split_data_train_test(image_dir,mask_dir,class_label,split_ratio):
    """
    This function tries to split the data to train and test according to split ratio
    """
    # Gather image-mask pairs and class presence
    class_ids = list(class_label.values())
    class_names = list(class_label.keys())

    data = []
    for fname in tqdm(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, fname)
        img_path = os.path.join(image_dir, fname)
        
        # Read mask in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Skip if image or mask is missing
        if mask is None or not os.path.exists(img_path):
            continue

        # Multi-hot vector for class presence
        class_vector = [int(cls_id in mask) for cls_id in class_ids]
        
        data.append((img_path, mask_path, *class_vector))

    # Build DataFrame
    columns = ['image_path', 'mask_path', 'MID', 'VTB', 'EPI']
    df = pd.DataFrame(data, columns=columns)

    # Extract features and labels
    X = df[['image_path', 'mask_path']]
    y = df[class_names].values

    # Split with Multi-label Stratification
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

    # Save or use the splits
    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    test_df.to_csv(TEST_SPLIT_CSV, index=False)

    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")

def create_train_validation_data(image_folder, annotation_path):
    """
    In this function, the train and validation images and masks are created using provided annotataions
    Final will have 2 csv's with details of train and test dataset.
    """

   # Read annotataion
    try:
        with open(annotation_path, 'r') as file:
            image_annotataion_json = json.load(file)

            
    except FileNotFoundError:
        print("Error: File not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    #create masks from annotataion
    create_folder(TEMP_MASK_FOLDER) #Create a temperory folder.
    for image,annotataions in image_annotataion_json.items():
        mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

        for annotataion in annotataions:
            class_label_value = CLASS_LABEL[annotataion[0]]
            annotataion_type = annotataion[1]
            point1 = (annotataion[2][0][0],annotataion[2][0][1])
            point2 = (annotataion[2][1][0],annotataion[2][1][1])

            if annotataion_type == "line":
                cv2.line(mask, point1, point2, class_label_value, thickness=10)
            elif annotataion_type == "rect":
                cv2.rectangle(mask, point1, point2, class_label_value, thickness=-1)

            else:
                print("This annotation type is unknown.")

        cv2.imwrite(TEMP_MASK_FOLDER+image+IMAGE_TYPE, mask) #Masks are created temperorly in this folder.
        print("Masks created for ", image)

    split_data_train_test(IMAGE_FOLDER,TEMP_MASK_FOLDER,CLASS_LABEL,TRAIN_TEST_SPLIT_RATIO)




#TESTING 
"""
Below is example on how to run this step.
"""
if __name__ == "__main__":
    create_train_validation_data(IMAGE_FOLDER,ANNOTATAION_PATH)

    # img = cv2.imread("./Dataset/masks/Image_97.png")
    # img = img * (255//3)
    # cv2.imwrite("sample_mask.png",img)

    



