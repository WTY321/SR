# SR: Weakly Supervised Chest X-ray Abnormality Localization

## Overview
Chest X-ray (CXR) imaging is widely used for diagnosing pulmonary diseases.  
To reduce the workload of radiologists and accelerate clinical interpretation, automated abnormality localization from chest X-rays has attracted increasing attention.

This repository implements a **weakly supervised object localization (WSOL)** framework for chest X-ray abnormality localization using **image-level labels only**, without requiring bounding box annotations.

The proposed framework adopts **VMamba** as the backbone and introduces three practical components to improve localization accuracy and robustness.

---

## Motivation
Most existing WSOL methods rely on **class activation maps (CAMs)** to localize abnormal regions.  
However, CAM-based approaches often suffer from:
- **Incomplete coverage** of abnormal regions  
- **Fragmented activation maps**
- Strong dependency on **classification confidence of specific categories**

These issues are particularly severe in chest X-ray images, where abnormalities are often subtle, diffuse, and low-contrast.

---

## Method
The proposed framework enhances weakly supervised abnormality localization by refining **Foreground Prediction Maps (FPMs)** through three key components.

### 1. Non-linear Modulation Module
We introduce a **non-linear modulation module** to:
- Expand discriminative foreground regions
- Improve spatial continuity of activations
- Reduce fragmented and sparse localization responses

This module refines the foreground prediction map by reshaping the activation distribution without relying on additional supervision.

### 2. FPM Fusion Module
An **FPM fusion module** is designed to:
- Enhance foreground responses
- Suppress background noise
- Increase foreground–background separability in chest X-ray images

This improves localization robustness under varying imaging conditions and disease appearances.

### 3. Foreground Control Loss
We propose a **foreground control loss** that:
- Adjusts feature activations to balance foreground and background
- Prevents foreground over-expansion
- Encourages more accurate abnormality localization

---

## Backbone Network
- **VMamba** is adopted as the backbone network for feature extraction.
- The implementation supports integration with other backbones (e.g., ResNet variants).

---

## Project Structure
SR/
├── DataLoader.py # Dataset definition and data loading pipeline
├── train.py # Training procedure
├── inference.py # Inference and localization generation
├── phototest.py # Visualization / testing utilities
├── Model/
│ ├── vmamba.py # VMamba backbone and model definition
│ ├── resnet.py # Optional ResNet backbone
│ ├── amm.py # Attention / modulation modules
│ └── csm_triton.py # Custom module implementation
├── utils/
│ ├── augment.py # Data augmentation
│ ├── accuracy.py # Classification accuracy
│ ├── IoU.py # Localization IoU evaluation
│ ├── util_cam.py # CAM / FPM utilities
│ ├── optimizer.py # Optimizer setup
│ ├── lr.py # Learning rate scheduling
│ ├── vis.py # Visualization tools
│ └── util.py # Common utility functions
## Dataset Structure
NIH/
├── train/
│ └── Atelectasis/
│ └── 00000011_006.png
├── test_list.txt
└── test_gt.txt
Training
python train.py


Training settings (e.g., backbone selection, learning rate, loss weights) can be modified inside train.py or corresponding configuration files.

Inference and Localization
python inference.py


The inference script generates foreground prediction maps and localization results from trained models.

Evaluation

Localization performance is evaluated using IoU-based metrics

Classification accuracy is also reported

Evaluation utilities are provided in utils/IoU.py and utils/accuracy.py

Reproducibility

To ensure reproducibility:

Fix random seeds

Use the same data splits as provided

Report backbone, training epochs, and optimizer settings

Notes

This repository focuses on weakly supervised localization, not fully supervised detection.

No bounding box annotations are used during training.

The framework is designed to be extensible to other medical imaging datasets.
