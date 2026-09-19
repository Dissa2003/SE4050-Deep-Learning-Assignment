# SE4050 Deep Learning Assignment - Brain Tumor Classification

## Overview

This repository contains the implementation for the **SE4050 - Deep Learning** university assignment. The project focuses on **Brain Tumor Classification** from MRI images, progressing from exploratory data analysis (EDA) through a classical machine learning baseline to custom deep learning models built with **PyTorch**.

## Problem Statement

Accurate and early detection of brain tumors from MRI scans is critical for effective diagnosis and treatment planning. Manual analysis is time-consuming and prone to human error. This project aims to build and evaluate deep learning models capable of automatically classifying MRI brain scans into tumor / non-tumor (or multi-class tumor types), comparing their performance against a traditional machine learning baseline.

## Folder Structure

```
se4050-deep-learning-assignment/
│
├── data/                          # Raw and processed datasets (ignored in git)
│
├── notebooks/                     # Jupyter notebooks for each project stage
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_baseline_model.ipynb    # Classical ML baseline model
│   └── 03_deep_learning_models.ipynb  # CNN / Deep Learning models (PyTorch)
│
├── src/                           # Reusable Python modules
│   ├── __init__.py
│   ├── dataset.py                 # Dataset loading & preprocessing utilities
│   ├── models.py                  # Model architecture definitions
│   └── utils.py                   # Training/evaluation helper functions
│
├── outputs/                       # Saved model checkpoints (.pth) and result plots
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment Setup & Installation Guide

### 1. Clone the Repository

```bash
git clone <repository-url>
cd se4050-deep-learning-assignment
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you have a CUDA-capable GPU, install the appropriate PyTorch build from the [official PyTorch website](https://pytorch.org/get-started/locally/) for GPU acceleration instead of the default CPU wheels.

### 4. Add the Dataset

Place the raw dataset inside the `data/` directory (this folder is git-ignored). Update dataset paths in `src/dataset.py` if required.

## How to Run the Project

1. **Exploratory Data Analysis** — Open and run `notebooks/01_eda.ipynb` to explore class distribution, sample images, and data quality.
2. **Baseline Model** — Run `notebooks/02_baseline_model.ipynb` to train and evaluate a classical machine learning baseline (e.g., SVM / Random Forest on extracted features).
3. **Deep Learning Models** — Run `notebooks/03_deep_learning_models.ipynb` to train and evaluate CNN-based deep learning models built using PyTorch.

Launch Jupyter to run the notebooks:

```bash
jupyter notebook
```

Trained model checkpoints and evaluation plots will be saved automatically to the `outputs/` directory.

## Author

SE4050 - Deep Learning Assignment
