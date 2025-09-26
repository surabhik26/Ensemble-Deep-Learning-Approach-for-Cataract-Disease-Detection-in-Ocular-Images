# Ensemble Deep Learning for Cataract Detection

## Project Overview
This project develops an **automated cataract detection system** using ensemble deep learning techniques. Cataracts are a leading cause of visual impairment, and early detection is crucial for effective treatment. The system uses pretrained convolutional neural networks (VGG19, ResNet50, and EfficientNetB0) and combines their predictions using a stacked ensemble meta-classifier to classify ocular fundus images as **Normal** or **Cataract**.  

**Key Achievements:**
- Achieved **98.17% validation accuracy**.
- Supports clinical decision-making by providing reliable and fast cataract detection.
- Uses transfer learning and meta-learning techniques to improve diagnostic performance.

---

## Features
- **Transfer Learning:** Leverages pretrained models (ImageNet) for faster training and better feature extraction.
- **Data Preprocessing:** Includes normalization and augmentation (rotation, flip, zoom, shift) to improve generalization.
- **Ensemble Learning:** Combines predictions from multiple models using a meta-classifier to enhance accuracy and robustness.
- **Binary Classification:** Predicts whether an eye is Normal or affected by Cataract.
- **Early Stopping:** Prevents overfitting by stopping training when validation performance plateaus.

---

## Technologies & Tools
- **Programming Language:** Python 3.x  
- **Deep Learning Frameworks:** TensorFlow / Keras, PyTorch (depending on implementation)  
- **Pretrained Models:** VGG19, ResNet50, EfficientNetB0  
- **Libraries:** NumPy, OpenCV, Matplotlib, scikit-learn, pandas  

---

## Dataset
- Ocular fundus images labeled as **Normal** or **Cataract**.  
- Dataset split into **Training** and **Validation** sets.  
- **Preprocessing includes:** normalization, resizing, and augmentation.
  https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

---

## Model Architecture
1. **Base Models:** VGG19, ResNet50, EfficientNetB0 (pretrained on ImageNet)  
2. **Modified Top Layer:** Dense layer with 2 neurons + Softmax activation for binary classification  
3. **Meta-Classifier:** Combines predictions from base models to produce final output  

**Training Details:**
- Optimizer: Adam  
- Loss Function: Binary Cross-Entropy  
- Metrics: Accuracy  
- Batch Size: 32–64  
- Epochs: with Early Stopping  
