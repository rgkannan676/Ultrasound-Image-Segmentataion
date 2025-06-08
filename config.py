#PATH DETAILS
"""
Contains global configuration variables such as data paths, class labels, and constants used across the project.
"""

ANNOTATAION_PATH = "./Dataset/annotations.json"
IMAGE_FOLDER = "./Dataset/images/"
TEMP_MASK_FOLDER = "./Dataset/masks/"
TRAIN_TEST_FOLDER = "./Train_Test_Dataset/."

TRAIN_SPLIT_CSV = "train_split.csv"
TEST_SPLIT_CSV = "test_split.csv"


CLASS_LABEL = {
    "MID":1,
    "VTB":2,
    "EPI":3
}

CLASS_SHAPE ={
    1:"LINE",
    2:"RECTANGLE",
    3:"LINE"
}

 #Class overlap : class that overlaps top
CLASS_OVERLAP_FIX = {
    1:3,
    2:1
}

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_TYPE=".png"

TRAIN_TEST_SPLIT_RATIO =0.2
NUMBER_OF_AUGMENTATAIONS = 7

TRAIN_EPOCHS = 500
VALIDATION_INTERVAL=5
#This is just for visulaization
SAVE_TRAIN_TEST_IMAGES = True

SAVE_MODEL_FOLDER = "./Models/"
LOG_DIR ="./log/"

INFERENCE_IMAGES_PATH = "./Dataset/images/" #  "./Dataset/test_image/" #
INFERENCE_RESULT_PATH = "./result/"
SAVE_INFERENCE_MASK = False
INFERENCE_MASKS_PATH = "./result_mask/"
INFERENCE_MODEL = SAVE_MODEL_FOLDER + "best_model.pth" # "best_model.pth" # 
