# Microrobot Depth and Pose Estimation

This repository contains the implementation of a deep learning-based framework for estimating the **depth** and **pose** (out-of-plane orientation and planar rotation angle) of microrobots from microscopic imaging data. Using AlexNet, ResNet50, and DenseNet169, this project evaluates the performance of different convolutional neural network architectures for accurate microrobot tracking tasks.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
   - [Original Dataset](#original-dataset)
   - [Preprocessing](#preprocessing)
   - [Self-generated Dataset](#self-generated-dataset)
5. [Training and Evaluation](#training-and-evaluation)
6. [Explainability with Grad-CAM](#explainability-with-grad-cam)
7. [Future Work](#future-work)
8. [Supplementary Material](#supplementary-material)

---

## Overview

Microrobots hold promise in biomedical fields such as minimally invasive surgery and targeted drug delivery. However, tracking their 3D position and orientation poses challenges due to:
- Noise in imaging
- Transparency of microrobots
- Limited labeled data in microscopic datasets

This project addresses these challenges by applying deep learning techniques for:
- **Depth Estimation**
- **Out-of-Plane Pose Estimation**
- **Planar Rotation Angle Estimation**

---

## Features

- Implementation of AlexNet, ResNet50, and DenseNet169 for regression and classification tasks.
- Grad-CAM integration for interpretability and model explainability.
- A robust preprocessing pipeline to handle noisy microscopic imaging data.
- Planar rotation augmentation to enrich the dataset and improve model generalization.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Eins51/Microrobot-Depth-and-Pose-Estimation.git
   cd Microrobot-Depth-and-Pose-Estimation
   ```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Ensure `PyTorch` and `CUDA` are properly configured for GPU-based training.

---

## Dataset Preparation

The dataset consists of cropped microrobot images categorized by out-of-plane poses and planar rotation angles. Below is an example of the structure after preprocessing:

### Original Dataset

The dataset consists of 36 subfolders named in the format `Pxx_Ryy`, where:
- `Pxx` denotes the **pitch angle** (e.g., `P0`, `P10`, ..., `P70`)
- `Ryy` represents the **roll angle** (e.g., `R0`, `R10`, ..., `R70`).

Each subfolder contains:
- **One `.avi` video file**, which includes between 600 and 1500 frames representing microrobot movements at a specific out-of-plane pose.
- **One `.txt` file**, recording the time frames and corresponding depth labels.

Below is an example of the dataset structure:

```
Dataset/

│     #  PXX_RYY, where PXX denotes the pitch angle and RYY represents the roll angle

├── P0_R0/ 

│   ├── 23_19_45_test.avi       # AVI video of microrobot at pitch=0°, roll=0°

│   ├── 23_19_45_data.txt       # Text file with timestamps and depth labels

│

├── P0_R10/

│   ├── 13_18_55_test.avi

│   ├── 13_18_55_data.txt

│

├── P10_R20/

│   ├── 16_07_36_test.avi

│   ├── 16_07_36_data.txt

│

⋮

└── P70_R70/

    ├── 23_38_15_test.avi

    └── 23_38_15_data.txt

```

### Preprocessing

##### *<u>1. Description</u>*

The preprocessing pipeline processes raw `.avi` video files of microrobots and prepares the dataset for depth and pose estimation tasks. This step involves:

1. **Frame Extraction**: Extracts individual frames from `.avi` video files and saves them as `.png` images.
2. **Gaussian Filtering**: Applies a Gaussian filter to remove noise and smoothen the frames.
3. **Binary Segmentation**: Performs binary segmentation using Otsu's thresholding combined with gradient magnitude to segment the microrobot.
4. **Contour Detection**: Identifies the largest contour corresponding to the microrobot's boundary.
5. **Centroid Calculation**: Calculates the intensity-weighted centroid of the segmented microrobot using the Gaussian-filtered frame.
6. **Image Cropping**: Crops a 230x230 region centered around the centroid to focus on the microrobot.
7. **Annotated Frames**: Saves annotated images with bounding boxes and labeled centroids for visual inspection.

##### *<u>2. Overview of Scripts</u>*

- **`preprocess.py`**: A script for processing raw video files, including frame extraction, contour detection, centroid calculation, and cropping.

##### *<u>3. Output Structure</u>*

For each `.avi` video file, the pipeline generates the following outputs in its corresponding subfolder:

```
<Root_Folder>/<Pxx_Ryy>/
├── output_images/                # Extracted frames from the video
├── contours_centroid/            # Images with contours and centroids annotated
│   └── centroid_coordinates.txt  # Text file with frame-wise centroid coordinates
├── cropped_images/               # Cropped microrobot images
└── annotated_images/             # Annotated frames with bounding boxes and centroids
```

### Self-Generated Dataset

##### *<u>1. Description</u>*

The **self-generated datasets** are created to augment the cropped microrobot images with planar rotations for improved training and evaluation of planar rotation angle estimation tasks. These datasets provide a comprehensive set of images for all rotation angles in 5° increments (0° to 355°), ensuring sufficient diversity for model training.

##### *<u>2. Overview of Scripts</u>*

- **`generate_dataset.py`**: This script generates a planar rotation angle dataset by rotating cropped microrobot images at 5° increments (0° to 355°), saving them in 72 subfolders (`Y0`, `Y5`, ..., `Y355`), and recording metadata in `.txt` and `.csv` files.

##### *<u>3. Output Structure</u>*

- **`Y0`, `Y5`, ..., `Y355`**: Contain rotated images corresponding to each angle.
- **`image_rotation_data.txt`**: Records the filenames and their respective rotation angles in a tab-separated format.
- **`image_rotation_data.csv`**: Records the same information in a CSV format for easier parsing.

```
planar_rotation_dataset/
	Y0/               # Images rotated by 0°
	Y5/               # Images rotated by 5°
	...
	Y355/             # Images rotated by 355°
	image_rotation_data.csv
	image_rotation_data.txt
```

## Training

### Overview of Scripts
- **`train.py`**: Used to train classification models for out-of-plane pose and planar rotation angle estimation tasks.
- **`train_regression.py`**: Used to train regression models for depth estimation.
- **`model.py`**: Defines the architectures for AlexNet, ResNet50, and DenseNet169.
- **`loss.py`**: Contains the loss functions used for classification and regression tasks.
- **`datasets.py`**: Handles dataset loading and preprocessing.
- **`split_dataset.py`**: Creates training and validation datasets by splitting processed data based on a specified ratio.
- **`txt_label.py`**: Exports depth information and generates labeled `.txt` files with centroid coordinates and associated depth values.
- - **`run.sh`**: Automates the process of training and evaluating models using pre-configured settings.
- **`configs/`**: Stores configuration files for different experiments.

### Running the Training Scripts
1. **Classification Model Training**:
   Run the following command to train a classification model:
   ```bash
   python train.py --config configs/pose_config.yaml
   ```

2. **Regression Model Training**:

   Run the following command to train a regression model:

```bash
python train_regression.py --config configs/depth_config.yaml
```

------

## Evaluation and Visualization

Evaluate trained models on the test dataset. The evaluation results, including metrics like RMSE, accuracy, and confusion matrices, will be saved in the `results/` folder.

### Overview of Scripts

- **`plot.py`**: Provides visualization methods for training and evaluation metrics, including `parse_log_and_plot` for classification tasks and `parse_log_and_plot2` for regression tasks.
- **`evaluate.py`**: Evaluates model performance and identifies bad cases for further analysis.
- **`gradcam.py`**: Generates Grad-CAM visualizations to interpret model feature activation and decision-making processes.

------

## Explainability with Grad-CAM

Grad-CAM provides insights into the decision-making process by highlighting regions of the input image that contribute most to the model’s predictions. This interpretability is crucial for deployment in biomedical applications where validation and reliability are paramount.

------

## Future Work

- **Unified Multi-Task Learning Framework**: Integrate depth estimation, pose classification, and rotation angle prediction into a single multi-task learning model to leverage shared feature representations.
- **Diverse Data Augmentation**: Expand the dataset to include more complex microrobot shapes and account for real-world imaging artifacts.
- **Real-World Testing**: Evaluate the framework on datasets with noisy and diverse imaging conditions to assess generalizability.
- **Explainability Integration**: Enhance the use of Grad-CAM to inform model improvement and assist biomedical decision-making.



------

## Supplementary Material

- **models_for_reference/**: Contains the implementation details of AlexNet, ResNet50, and DenseNet169 for microrobot depth and pose estimation.
- **plot_model_architectures/**: Includes visualizations of the implemented model architectures, generated using [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet).