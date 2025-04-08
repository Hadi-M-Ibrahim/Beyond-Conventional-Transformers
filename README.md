# Beyond Conventional Transformers: The Medical X-ray Attention (MXA) Block for Improved Multi-Label Diagnosis Using Knowledge Distillation

## Overview

This repository contains the implementation of the Medical X-ray Attention (MXA) block and its integration into Efficient Vision Transformers (EfficientViTs) for multi-label chest X-ray classification. The work is based on the paper **"Beyond Conventional Transformers: The Medical X-ray Attention (MXA) Block for Improved Multi-Label Diagnosis Using Knowledge Distillation"** by Hadi Ibrahim and Amit Rand. The repository includes the code for training, evaluation, and visualization of results on the CheXpert dataset.

## Features

- **Medical X-ray Attention (MXA) Block**: A novel attention mechanism tailored for medical imaging, combining Dynamic Region-of-Interest (ROI) Pooling and CBAM-style attention to capture localized and global features.
- **Efficient Vision Transformers (EfficientViTs)**: Adapted for multi-label classification with modifications to loss functions and architecture.
- **Knowledge Distillation (KD)**: Leveraging DenseNet-121 as a teacher model to improve student model performance under data constraints.
- **Multi-label Classification**: Designed for handling multiple co-occurring pathologies in chest X-rays.
- **CheXpert Dataset Integration**: Preprocessing, training, and evaluation pipelines for the CheXpert dataset.

## Acknowledgments
This repository is a fork of the Cream repository by Microsoft Research, with significant modifications for multi-label classification and medical imaging tasks. We also acknowledge the contributions of the CheXpert dataset team and the TorchXRayVision library.

## Citation

If our research was helpful to you, please cite the following paper [link](https://www.arxiv.org/abs/2504.02277):


```bibtex
@article{rand2025beyond,
  title={Beyond Conventional Transformers: The Medical X-ray Attention (MXA) Block for Improved Multi-Label Diagnosis Using Knowledge Distillation},
  author={Rand, Amit and Ibrahim, Hadi},
  journal={arXiv preprint arXiv:2504.02277},
  year={2025}
}

