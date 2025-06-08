# Ultrasound-Image-Segmentataion
This GitHub repository serves as a sample project demonstrating how to perform segmentation on ultrasound images using UNet and MONAI. Due to data privacy restrictions, the original dataset and trained models cannot be shared.This is not production-ready code, but rather an illustrative example of the workflow and methodology.
The project focuses on segmenting three anatomical structures from spinal ultrasound images:Vertebral Body (VTB), Epidural Space (EPI) and Midline (MID)

## ML Pilpeline
Implemented core components of the ML pipeline:  
- Train-test data splitting  
- Image preprocessing  
- Model training  
- Inference

## Data
The original dataset cannot be shared as it is private property. However, a sample dataset is included in the Datasets/ folder for demonstration purposes. Please note that these are not the actual images used in training. The sample annotations illustrate the representation format:
- VTB is marked as a rectangle  
- MID and EPI are annotated as lines

This sample is intended to show the annotation style and integration with the pipeline.

## Train-test data splitting
The dataset consists of 70 annotated ultrasound images, split into 80% training and 20% testing using MultilabelStratifiedShuffleSplit (from iterstrat) to ensure balanced distribution of MID, VTB, and EPI labels. Segmentation masks were generated after the split, with label mapping as follows: MID → 1, VTB → 2, EPI → 3. A line thickness of 10 pixels was used for MID and EPI to improve training stability. All images, masks, and split metadata (train_split.csv, test_split.csv) are saved to disk.
The script train_test_data_preparation.py automates the splitting, annotation parsing, and mask generation process.

## Image preprocessing
Image and mask paths are loaded from train_split.csv and test_split.csv.  
- Preprocessing includes:  
-- Noise Reduction: cv2.fastNlMeansDenoising preserves fine ultrasound structures  
-- Contrast Enhancement: CLAHE improves local contrast  
- Edge Enhancement: cv2.addWeighted sharpens anatomical boundaries  
- Training Augmentations:  
-- Horizontal Flip (50%)  
-- Shift, Scale, Rotate (±10°, 40% probability)  
-- Elastic Transform (20%) – simulates tissue deformation  
-- Grid Distortion (20%) – models probe pressure effects  
Each training image is augmented 7× to increase diversity. Test images undergo only preprocessing (no augmentation) for consistency. The image_preprocessing.py script applies these steps and saves outputs to ./Train_Test_Dataset/.

## Framework & Model Summary
The project uses MONAI (Medical Open Network for AI), a PyTorch-based framework optimized for medical imaging. MONAI offers:
- Pre-built segmentation models  
- Medical-specific transforms and data loaders  
- Flexible loss functions  

Model Chosen: UNet : UNet is well-suited for medical image segmentation, especially with small datasets. Its encoder-decoder architecture with skip connections enables:  
- Fine-grained boundary detection  
- Accurate localization  
- Multi-scale feature capture while preserving spatial detail
MONAI’s UNet provided the right balance of simplicity, robustness, and medical imaging suitability for grayscale ultrasound data—making it the most practical choice.

## Training Setup 
The model is trained using a PyTorch loop with the Adam optimizer for adaptive learning and fast convergence.
- Loss Function:  
-- DiceCELoss: Combines Dice Loss and Cross Entropy Loss to balance region overlap and pixel accuracy  
-- HausdorffDTLoss: Enhances boundary precision using distance transforms; applied after warm-up to prevent early instability  

- Training Strategy:  
-- Best checkpoint selected based on validation Dice score   
-- Training monitored via TensorBoard  
-- Final model saved to ./Models/  

- Best Epoch – Test Performance:  
  MID Dice Score: 0.5569  
  VTB Dice Score: 0.8185  
  EPI Dice Score: 0.7186  
  Average Dice Score: 0.698  

These results show strong performance, particularly for VTB and EPI segmentation.The training.py script handles training, evaluation, logging, and checkpointing.

## Inference
The pipeline uses the same preprocessing steps as training. The best-performing model is loaded to GPU for fast inference.
- Inference Steps:  
-- Input image → model → 4-channel probability map
-- Softmax + Argmax → pixel-wise class labels: Background, MID, VTB, EPI
- Post-Processing:  
-- Gap Fixing: Resolves gaps caused by structural overlaps : EPI over MID → MID gaps : MID over VTB → VTB gaps  
-- Anatomical Rule-Based Filtering: Suppresses VTB/EPI if MID is absent to Reduces false positives and Enforces anatomical plausibility.  
- Performance:  
-- Average Inference Time: 0.07 sec/image  
-- Tested on: NVIDIA GeForce GTX 1050 Ti  
The inference.py script processes a folder of images and saves output masks to the ./result/ directory.
