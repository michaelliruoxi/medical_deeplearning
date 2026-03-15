# COVID-19 Radiography Classification using Deep Learning

A deep learning project for classifying chest X-ray images into three categories — **COVID-19**, **Normal**, and **Viral Pneumonia** — using convolutional neural networks and transfer learning.

**GitHub**: https://github.com/lz3073-oss/P2_G4-Project_2.git

## Dataset

**Source**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

| Class | Original Count | Balanced Count |
|-------|---------------|----------------|
| COVID | 3,616 | 1,344 |
| Normal | 10,192 | 1,344 |
| Viral Pneumonia | 1,345 | 1,344 |
| **Total** | **15,153** | **4,032** |

The original dataset has a 7.6:1 class imbalance (Normal vs. Viral Pneumonia). Each class was capped at 1,344 images for balanced training. Images are resized to 192x192x3 and normalized to [0, 1].

**Split**: 85% train (3,427) / 15% test (605), stratified.

## Models

Six architectures were trained and compared:

| Rank | Model | Type | Test Accuracy | AUC |
|------|-------|------|---------------|-----|
| 1 | **CNN + Augmentation** | Custom | **95.70%** | 0.9942 |
| 2 | Baseline CNN | Custom | 95.04% | 0.9917 |
| 3 | ResNet50 | Transfer | 91.90% | — |
| 4 | VGG16 | Transfer | ~92% | — |
| 5 | InceptionV3 | Transfer | ~87% | — |
| 6 | Deeper CNN (4-block) | Custom | ~87% | — |

### Baseline CNN
Sequential CNN with 3 convolutional blocks (32 → 64 → 128 filters), BatchNorm, MaxPooling, followed by Dense(256) + Dropout(0.4). Trained with Adam (lr=1e-4), early stopping (patience=6), for up to 30 epochs.

### CNN + Data Augmentation (Best Model)
Same architecture as baseline, trained with `ImageDataGenerator` applying random rotation, width/height shifts, zoom, and horizontal flips. Achieved +0.66% improvement over the baseline.

### Transfer Learning Models
ResNet50, VGG16, and InceptionV3 pre-trained on ImageNet with frozen base layers, a custom classification head (GlobalAveragePooling2D → Dense → Dropout → Dense(3, softmax)), and optional fine-tuning of top layers. Transfer learning underperformed custom CNNs due to domain mismatch between ImageNet natural images and grayscale medical X-rays.

## Key Findings

- **Custom CNNs outperformed transfer learning** for this medical imaging task — the domain gap between ImageNet and chest X-rays reduced the benefit of pre-trained features.
- **Data augmentation** provided meaningful regularization, boosting accuracy from 95.04% to 95.70%.
- The best model achieved per-class F1 scores of 0.94 (COVID), 0.94 (Normal), and 0.99 (Viral Pneumonia).

## Setup

### Requirements

```
numpy<2
ml-dtypes<0.5
opencv-python-headless<4.9
pandas<2.3
scipy<1.12
scikit-learn<1.5
matplotlib
seaborn
jupyter
tensorflow
```

### Installation

```bash
pip install -r requirements_container.txt
pip install tensorflow
```

### Running

```bash
jupyter notebook Project_2.ipynb
```

The notebook expects the `COVID-19_Radiography_Dataset/` directory in the project root with subdirectories `COVID/images/`, `Normal/images/`, and `Viral Pneumonia/images/`.

## Project Structure

```
├── Project_2.ipynb              # Main notebook (data processing, training, evaluation)
├── requirements_container.txt   # Python dependencies
├── README.md
└── COVID-19_Radiography_Dataset/
    ├── COVID/images/
    ├── Normal/images/
    └── Viral Pneumonia/images/
```
