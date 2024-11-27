# Handwritten Character Recognition

## Project Overview
This project focuses on building a deep learning-based system capable of recognizing handwritten characters. Using a 3-layer neural network, the system is trained on the MNIST dataset to recognize digits (0-9) and evaluated on a custom dataset of handwritten characters created manually. The project explores key concepts in data preprocessing, model training, and evaluation while addressing challenges related to handwriting variability.

## Dataset Description

1. **Training Dataset**:
   - **Source**: MNIST Dataset.
   - **Content**:
     - \(60,000\) training images and \(10,000\) testing images of digits (0-9).
     - Each image is \(28 \times 28\) pixels, grayscale, with white digits on a black background.

2. **Custom Testing Dataset**:
   - Created manually by writing digits \(0-9\) on paper, capturing images, and preprocessing them to match MNIST format.
   - **Preprocessing Steps**:
     - Converted to grayscale.
     - Resized to \(28 \times 28\) pixels.
     - Normalized pixel values to the range \([0, 1]\).
     - Inverted colors for consistency with MNIST images.

---

## Project Objectives
- Train a 3-layer deep neural network on the MNIST dataset for digit recognition.
- Test the trained model on custom handwritten digit images.
- Evaluate model performance and analyze misclassifications.
- Explore techniques to improve model generalization on handwritten data.

---

## Model Architecture

### 1. Input Layer
- **Function**: Converts \(28 \times 28\) images into 1D arrays of 784 pixels.

### 2. Hidden Layers
- **Layer 1**:
  - 128 neurons, ReLU activation function.
  - Extracts initial features such as edges and curves.
- **Layer 2**:
  - 64 neurons, ReLU activation function.
  - Focuses on higher-level patterns like loops and digit structures.

### 3. Output Layer
- 10 neurons (one for each digit: 0-9).
- **Activation Function**: Softmax for generating class probabilities.

### Optimizer: Adam (adaptive learning rate for efficient convergence).  
### Loss Function: Categorical Cross-Entropy (multi-class classification).  
### Evaluation Metric: Accuracy.

---

## Key Features
1. **Custom Handwritten Digit Dataset**:
   - Personal handwritten samples provide a real-world challenge for the model.

2. **Data Preprocessing**:
   - Ensures consistency between training (MNIST) and testing datasets.

3. **Visualization**:
   - Displays detected and undetected digits, helping identify patterns in misclassifications.

4. **Performance Analysis**:
   - Reports accuracy and shows which digits were misclassified.

---

## Installation and Setup

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

### Installation
Clone the repository:

   ```bash
   git clone https://github.com/your-username/handwritten-character-recognition.git
  ```

### Thank you!
