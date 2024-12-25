import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from COVID19Dataset import COVID19Dataset

dataset_path = "datasets/COVID-19_Radiography_Dataset"

def confusion_matrix(y, y_pred, num_classes):
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true_class, pred_class in zip(y, y_pred):
        confusion_matrix[true_class, pred_class] += 1
    return confusion_matrix

def display_confusion_matrix(confusion_matrix,num_classes):
    confusion_matrix_np = confusion_matrix.numpy()
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_np, display_labels=range(num_classes))
    confusion_matrix_display.plot()
    plt.show()
    print("Confusion Matrix: \n", confusion_matrix.numpy())

def display_train(train_dataloader):
    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

def get_dataset(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return COVID19Dataset(root_dir=root_dir, transform=transform)

if __name__ == '__main__':
    y = torch.tensor(np.random.randint(0, 3, 1000))
    y_pred = torch.tensor(np.random.randint(0, 3, 1000))
    num_classes = 3
    confusion_matrix = confusion_matrix(y, y_pred, num_classes)
    display_confusion_matrix(confusion_matrix, num_classes)
    dataset = get_dataset(dataset_path)
    training_data, test_data = train_test_split(dataset, test_size=0.3, shuffle=True)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # display_train(train_dataloader)
