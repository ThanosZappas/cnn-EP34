from collections import Counter

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
DATA_ROOT_DIR = "datasets/Trimmed_COVID-19_Radiography_Dataset"


def train_val_test_split(dataset, lengths):
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, lengths, generator=generator)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dataloaders(train_ds, val_ds, test_ds, batch_size, shuffle=True):
    return DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle), \
        DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle), \
        DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)


def get_confusion_matrix(y, y_pred, num_classes):
    # Initialize an empty confusion matrix
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    # Populate the confusion matrix
    for true_class, pred_class in zip(y, y_pred):  # Ensure compatibility with tensors on GPU
        confusion_matrix[true_class, pred_class] += 1

    return confusion_matrix


def display_confusion_matrix(confusion_matrix):
    # Convert to NumPy for visualization
    confusion_matrix = confusion_matrix.numpy()

    # Display the confusion matrix
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=CLASSES)
    confusion_matrix_display.plot()
    plt.show()

    print("Confusion Matrix: \n", confusion_matrix)


def get_dataset(root_dir):
    from COVID19Dataset import COVID19Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return COVID19Dataset(root_dir=root_dir, transform=transform)


def display_batch(dataset, random_indexes):
    dataset.display_batch(random_indexes)
    label_counts = Counter(dataset.labels)
    classes = dataset.classes
    counts = [label_counts[i] for i in range(len(classes))]

    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Class')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    for cls, count in zip(classes, counts):
        print(f"Class: {cls}, Number of Images: {count}")


import os
import shutil
import random


def trim_dataset(original_dir, target_dir, class_ratios):
    """
    Trims the dataset dynamically according to specified class ratios.

    Args:
        original_dir (str): Path to the original dataset directory.
        target_dir (str): Path to save the trimmed dataset.
        class_ratios (dict): Ratios for each class as a dictionary with class names as keys
                             and desired ratios as values, e.g., {'COVID': 0.1, 'Normal': 0.2}.
    """
    # If the target directory already exists, delete it
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir)

    for class_name, ratio in class_ratios.items():
        original_class_dir = os.path.join(original_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)

        if not os.path.exists(original_class_dir):
            print(f"Class directory {original_class_dir} does not exist. Skipping.")
            continue

        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        # Get all image files in the class directory
        all_images = [f for f in os.listdir(original_class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(all_images)

        # Determine the number of images to copy based on the ratio
        num_to_copy = int(total_images * ratio)

        # Randomly select images to copy
        selected_images = random.sample(all_images, num_to_copy)

        # Copy selected images to the target directory
        for img in selected_images:
            shutil.copy(os.path.join(original_class_dir, img), os.path.join(target_class_dir, img))

        print(f"Copied {num_to_copy} images from {class_name} to {target_class_dir}.")
    print("Dataset trimming complete!")


def make_small_dataset():
    original_dataset_path = "datasets/COVID-19_Radiography_Dataset"
    trimmed_dataset_path = "datasets/Trimmed_COVID-19_Radiography_Dataset"
    class_ratios = {'COVID': 0.05, 'Lung_Opacity': 0.05, 'Normal': 0.05, 'Viral Pneumonia': 0.05}

    trim_dataset(original_dataset_path, trimmed_dataset_path, class_ratios)

# if __name__ == '__main__':
# make_small_dataset()
