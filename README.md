

# Image Classification with Support Vector Machine 


This repository contains a Python script for image classification using Support Vector Machine (SVM). The script loads a custom image dataset, preprocesses the images, trains an SVM model, and evaluates its performance using various classification metrics.

## Features
Loads a custom image dataset
Splits the dataset into training and testing sets
Preprocesses images (resizing, flattening)
Trains an SVM model on the training set
Evaluates the model's performance using accuracy, precision, recall, F1-score, and confusion matrix
Visualizes classification metrics in 3D space

## Requirements
Python 3.x
OpenCV
NumPy
Matplotlib
scikit-learn
Seaborn
## Usage
### Clone the repository:

https://github.com/Tanmay-Butta/SVMImageClassifier

### Install the required dependencies:


pip install opencv-python numpy matplotlib scikit-learn seaborn

### Prepare your image dataset
 update the directory path in the script where images are loaded (dir variable).

### Run the script:
python script.py

### View the results 
including accuracy, classification report, confusion matrix, and a 3D visualization of classification metrics.

## Files
script.py: Main Python script for image classification.

data1.pickle: Pickle file containing the preprocessed image dataset.

model3.sav: Pickle file containing the trained SVM model.

## Acknowledgments
The dataset used in this project was sourced from ImageNet.



