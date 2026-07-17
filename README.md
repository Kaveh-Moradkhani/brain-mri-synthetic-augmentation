
# Brain MRI Synthetic Augmentation — Neuromatch Academy Project

## Project overview
This project investigates deep learning approaches for brain MRI analysis using FLAIR images. We focus on three complementary tasks:
- Brain lesion segmentation using U-Net-based architectures.
- Slice-level classification to identify whether an MRI slice contains lesions.
- Synthetic image generation using diffusion models as a potential data augmentation strategy to improve segmentation performance when labeled data are limited.
Since diffusion models require substantial computational resources, the segmentation and classification pipelines are being developed in parallel while synthetic augmentation experiments are ongoing.

## Objectives
Our project aims to:
- Preprocess raw brain MRI scans into training-ready datasets.
- Train a lesion segmentation model.
- Develop a binary classifier that detects lesion-containing slices.
- Evaluate whether synthetic MRI images can improve segmentation performance under limited-data settings.

## Team workflow
- GitHub: code, notebooks, configs, reports, small results
- Google Drive: dataset, generated images, checkpoints, large outputs
- Google Colab: training and experiments
