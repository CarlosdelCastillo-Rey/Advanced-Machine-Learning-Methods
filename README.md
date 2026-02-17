# Advanced-Machine-Learning-Methods

## Authors

- **Carlos Fernando Del Castillo Rey** — A01796595  
- **Uriel Derrant Soriano** — A01373541  
- **Julio Miguel Díaz de León García** — A01796215  
- **Luis Ángel Díaz Rebollo**


## Activities Overview

This repository contains the implementation and analysis of two related activities focused on image classification using PyTorch and the CIFAR-10 dataset:

- **Activity 2b: Building a CNN for CIFAR-10 Dataset with PyTorch**
- **Activity 2c: Exploring Transfer Learning with CIFAR-10**

Together, these activities explore the progression from simple baseline models to convolutional neural networks and finally to transfer learning using pre-trained architectures.

---

## Dataset: CIFAR-10

CIFAR-10 is a standard benchmark dataset composed of **60,000 RGB images** of size **32×32 pixels**, evenly distributed across **10 classes**:

- airplane  
- automobile  
- bird  
- cat  
- deer  
- dog  
- frog  
- horse  
- ship  
- truck  

The dataset is split into:
- 50,000 training images (from which a validation subset is derived)
- 10,000 test images

All images are normalized channel-wise prior to training.

---

## Activity 2b: Building a CNN for CIFAR-10 Dataset with PyTorch

### Objective

The objective of this activity is to build and analyze image classification models from scratch using PyTorch, starting with a simple baseline and progressing to a convolutional neural network.

---

### Linear Classifier Baseline

A **linear classifier (softmax regression)** operating on flattened pixel values is trained as an initial baseline.

**Characteristics:**
- No spatial feature extraction
- Fast convergence
- Limited representational capacity

This model serves as a lower bound for performance and highlights the limitations of pixel-based representations in image classification tasks.

---

### Convolutional Neural Network (CNN)

A CNN is implemented to leverage the spatial structure of images.

**Key components:**
- Convolutional layers for local feature extraction
- Pooling layers for spatial downsampling
- Fully connected layers for classification

The CNN demonstrates significantly improved performance over the linear baseline, achieving higher training, validation, and test accuracy.

---

### Model Comparison and Analysis

The learning behavior of both models is analyzed using:
- Training accuracy per epoch
- Validation accuracy per epoch
- Validation loss per epoch
- Train–validation accuracy gap

These analyses illustrate:
- Early saturation of the linear classifier
- Progressive and stable learning of the CNN
- The importance of convolutional inductive bias for visual data

---

## Activity 2c: Exploring Transfer Learning with CIFAR-10

### Objective

This activity focuses on **Transfer Learning**, a powerful technique that improves model performance by leveraging **pre-trained architectures**.

A provided notebook demonstrates a complete solution using a specific pre-trained model on CIFAR-10.  
The task is extended by experimenting with **two additional pre-trained models**.

---

### Transfer Learning Methodology

The general workflow followed in this activity is:

1. Load a model pre-trained on a large-scale dataset (e.g., ImageNet).
2. Replace the final classification layer to match the CIFAR-10 classes.
3. Freeze the backbone network or selectively fine-tune higher layers.
4. Train the modified model on CIFAR-10.
5. Evaluate and compare performance against models trained from scratch.

---

### Expected Benefits of Transfer Learning

Transfer learning typically provides:
- Faster convergence
- Improved generalization
- Higher validation and test accuracy with fewer training epochs

This activity demonstrates how learned representations from large datasets can be effectively reused for smaller-scale image classification tasks.

---

## Technologies Used

- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  

---

 — A01365832  

---

## Conclusion

Across both activities, the experiments demonstrate the impact of model architecture and prior knowledge on image classification performance:

- Linear models provide simple and interpretable baselines but are limited in capacity.
- CNNs significantly improve performance by exploiting spatial structure.
- Transfer learning further enhances results by reusing pre-trained visual representations.

These activities collectively illustrate modern deep learning practices for computer vision using PyTorch.
