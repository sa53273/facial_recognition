# Facial Recognition with CNNs and Transfer Learning

## Overview

This project focuses on developing and evaluating machine learning models for facial recognition using the Labeled Faces in the Wild (LFW) dataset. The project involves implementing custom convolutional neural networks (CNNs) and transfer learning techniques to classify faces into predefined categories.

## Project Motivation

Facial recognition is a challenging task in computer vision due to variations in lighting, pose, expression, and occlusion. This project aims to explore and compare different deep learning approaches for facial recognition using the LFW dataset, which contains images of faces collected from the web.

## Dataset

The Labeled Faces in the Wild (LFW) dataset is a public benchmark for face verification and recognition tasks. It contains over 13,000 images of faces from various celebrities and public figures, each labeled with the name of the person pictured. For this project, the dataset is used for multi-class classification.

## Model Architectures

### Custom CNN Models

Two custom CNN architectures were implemented from scratch to classify the LFW dataset:

1. **CNN1:**
   - 3 convolutional layers with ReLU activation and max pooling.
   - Dense layers with dropout and L2 regularization.
   - Used for initial exploration of model performance on the dataset.

2. **CNN2:**
   - 4 convolutional layers with batch normalization.
   - Global average pooling followed by dense layers.
   - Includes additional regularization techniques for improved generalization.

### Transfer Learning with Pre-trained Models

Pre-trained models like ResNet50 and VGGFace were used to leverage the power of transfer learning. The models were fine-tuned on the LFW dataset after freezing the initial layers.

- **ResNet50:** Pre-trained on ImageNet, modified for grayscale images.
- **VGGFace:** Specifically designed for face recognition tasks.

## Training and Evaluation

### Class Weights

To address the class imbalance in the dataset, class weights were computed and incorporated into the training process. This ensures that the model doesn't become biased towards more frequent classes.

### Performance Metrics

The models were evaluated based on accuracy, loss, and confusion matrices. These metrics provided insights into the strengths and weaknesses of each approach.
