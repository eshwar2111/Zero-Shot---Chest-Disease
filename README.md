# Zero-Shot Learning (ZSL) for Medical Image Classification

## Overview
This project implements a **Zero-Shot Learning (ZSL) Human Disease Classifier** that utilizes **BERT and a custom CNN architecture** to classify **chest X-ray images**. The model is designed for **multilabel classification**, identifying various diseases from medical images.

## Dataset
- **CSV File**: `Data_Entry_2017.csv` (contains metadata about images and labels)
- **Image Directory**: `/kaggle/input/zsl-dataset4` (contains medical images)
- **Labels**:
  - Atelectasis
  - Cardiomegaly
  - Consolidation
  - Edema
  - Effusion
  - Emphysema
  - Fibrosis
  - Hernia
  - Infiltration
  - Mass
  - No Finding
  - Nodule
  - Pleural Thickening
  - Pneumonia
  - Pneumothorax

## Model Architecture
The model consists of a **custom CNN (Convolutional Neural Network)** with:
- **Two convolutional layers**
- **Max-pooling layers**
- **Fully connected layers**
- **Sigmoid activation for multilabel classification**

## Training Process
1. **Data Preprocessing**:
   - Load images from the dataset
   - Apply transformations (resize, convert to grayscale, normalize)
   - Convert multilabel disease classification into binary vectors
2. **Training Loop**:
   - Use **Binary Cross-Entropy Loss (BCELoss)**
   - Optimize using **Adam Optimizer**
   - Train for **5 epochs** on a **GPU-enabled Kaggle Notebook**

## Performance
- Loss values over 5 epochs:
  - Epoch 1: **0.2752**
  - Epoch 2: **0.2491**
  - Epoch 3: **0.2366**
  - Epoch 4: **0.2258**
  - Epoch 5: **0.2143**
- **Training completed successfully**

## How to Use
1. **Install dependencies**:
   ```bash
   pip install torch torchvision pandas pillow
   ```
2. **Run the model**:
   ```python
   python train.py
   ```
3. **Make predictions**:
   ```python
   from model import SimpleCNN
   from predict import predict_disease
   predict_disease(image_path='sample_image.png')
   ```



## Future Enhancements
- Integrate **BERT embeddings** for label representations
- Implement **Zero-Shot Learning (ZSL) with CLIP or Transformer models**
- Optimize performance for **real-time clinical applications**

## Contributors
- **Eshwar B.**

## License
MIT License

