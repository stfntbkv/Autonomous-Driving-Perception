# Autonomous Driving Perception - Object Detection and Drivable Area and Lane Markings Segmentation for Urban Scenes

**A complete deep learning pipeline for object detection and segmentation using BDD100K and Cityscapes datasets, with custom preprocessing, augmentations, and model training using PyTorch and Ultralytics YOLOv8.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Methodology](#methodology)
    * [Datasets](#datasets)
    * [Data Preparation](#data-preparation)
    * [Exploratory Data Analysis](#exploratory-data-analysis)
    * [Preprocessing & Augmentations](#preprocessing--augmentations)
    * [Model Architectures](#model-architectures)
* [Project Structure](#project-structure)
* [Setup and Installation](#setup-and-installation)
* [Results](#results)
* [Screenshots](#screenshots)

---

## Overview

This project develops perception models for autonomous driving, focusing on object detection and lane/drivable area segmentation. 
Using BDD100k and Cityscapes, we create custom segmentation masks and bounding boxes, preprocess data with extensive augmentations, and custom training two models with modified structures for our task: 
YOLOv8 for object detection and HRNet with two segmentation heads. The pipeline includes data preparation, exploratory data analysis (EDA), data preprocessing weighted sampling, and class-weighted loss functions to handle class imbalance, with evaluation on diverse driving conditions.

---

## Features

* **Custom Dataset Creation**: Transformed BDD100k polygonal annotations into segmentation masks for drivable areas and lane markings, and Cityscapes annotations into bounding boxes for object detection.

* **Comprehensive EDA**: Analyzed object, area, and image attribute distributions (e.g., weather, time of day) to identify and address rare scenarios and invalid labels.

* **Advanced Preprocessing**: Removed low-frequency classes, applied modified augmentations (Mosaic, Rotation, Translation etc.) from Ultralytics and Albumentations, and created weighted samplers for balanced training.

* **Dual-Model Architecture**:

  * Object Detection: YOLOv8 with custom class configuration and weighted loss function, trained for 100 epochs with AdamW and CosineAnnealingLR.
  * Segmentation: HRNet with dual heads for lane and drivable area segmentation, using pixel-frequency-weighted loss.

* **Robust Evaluation**: Assessed performance with mAP (detection) and IoU (segmentation) on full test sets and rare scenarios (e.g., night, rain) using TensorBoard logging.

* **PyTorch Integration**: Seamless Dataset and DataLoader implementation for efficient training and validation.

---

## Methodology

### Datasets

#### BDD100K
- Tasks used:
  - Drivable area segmentation
  - Lane marking segmentation
  - Object detection
- Created masks manually from polygon annotations
- Used official val/test splits for validation and testing

#### Cityscapes
- Tasks used:
  - Object detection only
- Converted polygonal annotations to bounding boxes

---

### Data Preparation

- Loaded images and labels, verified consistency
- Generated segmentation masks for BDD100K
- Converted polygonal annotations for Cityscapes

---

### Exploratory Data Analysis

- Object class distributions
- Polygonal area stats
- Weather/time-of-day/scene analysis
- Rare conditions flagged for evaluation
- Size and spatial distribution of objects
- Removed invalid boxes and unknown labels

---

### Preprocessing & Augmentations

- Removed underrepresented classes
- Augmentations included:
  - Resize, Mosaic
  - Rotation, Translation, Scaling
  - Shear, Perspective
  - Hue/Saturation/Value
  - Blur, Normalize, Horizontal Flip
- Unified all data and custom augmentations into custom PyTorch datasets
- Weighted sampling based on image-level attributes and objects
- Created separate loaders for train/val/test

---

### Model Architectures

#### Object Detection
- **YOLOv8-L (Ultralytics)**
- Custom class count via YAML
- Modified loss function to handle class weights
- Evaluation:
  - Full test set
  - Rare conditions subset
  - mAP and class-wise mAP scores

#### Segmentation
- **HRNet Backbone**
- Dual segmentation heads:
  - Lane markings
  - Drivable area
- Loss weighted by pixel frequency
- Evaluation:
  - IoU overall
  - IoU per class
  - Per-condition results

---

## Project Structure

```
```

## Setup and Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/stfntbkv/Object-Area-and-Lane-Detection-.git
   cd Object-Area-and-Lane-Detection-
   ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    # Ensure you have a compatible Python 3.x version installed
    python -m venv venv
    # Windows
    # venv\Scripts\activate
    # macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This will install all packages needed to run the project)*

4.  **Obtain Datasets:**
    * Download the BDD100k - images, labels and drivable areas and place the files inside the `data/` directory, the created segmentation you can download from here or create them yourself running the notebook.
    * Download the Cityscapes labels and images and place it in the `data/` directory.

5.  **Get Pre-trained Models:**
    * The pre-trained models required to run the web application are hosted on Google Drive.
    * **Download the model files from:**
        * [Pre-trained models](https://drive.google.com/drive/folders/19-hgxOFR1LY6Q3jrR2njuZvkr1S2hUwG?usp=share_link)

    * **Place the downloaded file(s)** inside the `saved_models/` directory in your project folder.

## **Results**

### **Detection Model**

**Overall Test Metrics**

| Metric / Subset            | Overall mAP | mAP@50 (IoU=0.5) | mAP@75 (IoU=0.75) |
|----------------------------|-------------|------------------|-------------------|
| **Full Test Set**          | 0.319       | 0.573            | 0.302             |
| Foggy                      | 0.349       | 0.595            | 0.312             |
| Snowy                      | 0.306       | 0.539            | 0.291             |
| Rainy                      | 0.310       | 0.575            | 0.288             |
| Undefined Weather          | 0.344       | 0.606            | 0.330             |
| Nighttime                  | 0.286       | 0.533            | 0.258             |
| Undefined Time of Day      | 0.486       | 0.749            | 0.401             |

**Per class Test Metrics**

| Class         | mAP   | mAP@50 | mAP@75 |
|---------------|-------|--------|--------|
| Car           | 0.450 | 0.723  | 0.460  |
| Traffic Sign  | 0.343 | 0.639  | 0.325  |
| Traffic Light | 0.222 | 0.587  | 0.123  |
| Person        | 0.312 | 0.630  | 0.272  |
| Truck         | 0.443 | 0.613  | 0.494  |
| Bus           | 0.411 | 0.540  | 0.462  |
| Bike          | 0.231 | 0.488  | 0.194  |
| Rider         | 0.235 | 0.467  | 0.216  |
| Motor         | 0.221 | 0.469  | 0.174  |

### **Segmentation Model**

