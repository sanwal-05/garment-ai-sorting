# AI-Assisted Sorting and Refurbishment of Returned Garments Using Multi-Task Deep Learning

## Overview

This project explores the use of multi-task deep learning to assess second-hand garments using front and back images. It predicts four garment attributes: condition, pilling, stains, and holes.

The predicted attributes can support a transparent, rule-based recommendation for the garment's potential next life, such as resale, refurbishment, repair, or recycling.

## Model

- Backbone: EfficientNetB0 pretrained on ImageNet
- Input: Front and back garment images
- Tasks: Condition, pilling, stains, and holes classification
- Framework: TensorFlow / Keras
- Evaluation: Accuracy, macro-F1, weighted-F1, and confusion matrices

## Repository Contents

- `DL_Proj_Improved.ipynb` — improved notebook and experiment analysis
- `run_experiments.py` — controlled experiment runner
- `extract_features.py` — feature extraction and caching
- `download_dataset.py` — dataset download and preparation
- `results/` — experiment metrics and evaluation outputs
- `figures/` — plots and confusion matrices

## Dataset

The project uses the Second-Hand Fashion Dataset v2 from Zenodo:

https://zenodo.org/records/12518734

The dataset is not included in this repository. Refer to the dataset's license and instructions before downloading or using it.

## Reproducibility

The dataset, cached features, and trained model checkpoints are excluded because of their size. Run the scripts only after setting up the required dependencies and preparing the dataset paths for your machine.

## Limitations

The dataset represents second-hand fashion sorting data and is used as a proxy for returned-garment assessment. Performance varies across prediction tasks, particularly for underrepresented defect classes. The recommendations are decision-support suggestions, not verified refurbishment decisions.

## Authors

Sanjana Bhowal, Riya Tyagi, and project collaborators.
