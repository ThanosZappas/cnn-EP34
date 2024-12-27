import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

import utils

CLASSES = utils.CLASSES
DATA_ROOT_DIR = utils.DATA_ROOT_DIR


class COVID19Dataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.labels = []
        self.classes = []

        # Filter only directories first
        valid_dirs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]

        # Enumerate over valid directories to ensure sequential label indices
        for label_idx, label_name in enumerate(valid_dirs):
            class_dir = os.path.join(root_dir, label_name)
            self.classes.append(label_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(file_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def display_batch(self, indexes):
        n = len(indexes)
        cols = int(n ** 0.5)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()

        for ax, idx in zip(axes, indexes):
            image, label = self[idx]

            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()  # (C, H, W) to (H, W, C)

            # Decode one-hot label for display if applicable
            if isinstance(label, torch.Tensor) and label.dim() == 1 and label.sum() == 1:
                label = label.argmax().item()

            ax.imshow(image)
            ax.set_title(self.classes[label])
            ax.axis('off')

        for ax in axes[len(indexes):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    dataset = utils.get_dataset(DATA_ROOT_DIR)

    random_indexes = random.sample(range(len(dataset)), 25)

    utils.display_batch(dataset, random_indexes)
