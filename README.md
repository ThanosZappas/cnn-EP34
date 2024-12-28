# cnn-EP34
https://eclass.hua.gr/modules/work/?course=DIT232&get=5788&file_type=1

# COVID-19 Radiography Classification

This project aims to classify COVID-19 radiography images using convolutional neural networks (CNNs). The dataset used includes images of COVID, Lung Opacity, Normal, and Viral Pneumonia cases.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ThanosZappas/covid19-radiography-classification.git
    cd covid19-radiography-classification
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the dataset:
    - Ensure the dataset is placed in the `datasets/COVID-19_Radiography_Dataset` directory.
    - Optionally, trim the dataset using the `make_small_dataset` function in `utils.py`.

2. Train the model:
    ```sh
    python train_test.py
    ```
   - If your computer isn't fit for the whole dataset, you can train with smaller parameters. Enter y/n when asked.
   - You will be asked to enter the model name you want to use.
   - Enter  CNN1 or  CNN2 or ResNet50 or Processed_ResNet50


3. Evaluate the model:
    - The script will automatically evaluate the model on the test set after training.

## Project Structure

- `classes/`: Contains the model definitions and utility classes.
- `datasets/`: Directory where the dataset should be placed.
- `utils.py`: Utility functions for data processing, splitting, and visualization.
- `train_test.py`: Main script for training and evaluating the model.

## Model Training

The model is trained using the following parameters:

### CNN Models
1. 'CNN1'
2. 'CNN2'
- Optimizer: Adam with learning rate \(1 \times 10^{-3}\), \(\beta_1 = 0.9\), \(\beta_2 = 0.99\)
- Loss Function: Cross-Entropy Loss
- Batch Size: 64
- Maximum Epochs: 20
- Early Stopping: Triggered if validation loss does not improve for 5 consecutive epochs

### ResNet Models
1. 'ResNet50' all layers are trainable
2. 'Processed_ResNet50' only the last layer is trainable
- Optimizer: Adam with learning rate \(1 \times 10^{-4}\), \(\beta_1 = 0.9\), \(\beta_2 = 0.99\)
- Loss Function: Cross-Entropy Loss
- Batch Size: 64
- Maximum Epochs: 5
- Early Stopping: Triggered if validation loss does not improve for 5 consecutive epochs

## Evaluation

The model is evaluated on the test set, and the following metrics are reported:
- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Confusion Matrix

## Results

The final results of the model training and evaluation will be printed in the console, including the accuracies and the confusion matrix.

## Example Usage
```sh
Do you have lower computer specifications? (y/n): y \
Running with optimized parameters for lower specifications.
Creating a smaller version of the dataset...

Using device: cpu

Available Models:
1. CNN1
2. CNN2
3. ResNet50
4. Processed ResNet50
Enter the number corresponding to the model you want to use (1-4): 2

Starting training...

Epoch 1
-------------------------------
Training Loss: 0.563892, Accuracy: 76.45%
Validation Loss: 0.483765, Accuracy: 82.31%

...

Training Summary:
Final Training Accuracy: 92.15%
Final Validation Accuracy: 89.34%

Evaluating on Test Set...
Accuracy: 90.1%, Avg loss: 0.425384
```