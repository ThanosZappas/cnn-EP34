
import torch.nn as nn


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 54 * 54, 32), # 16 * 54 * 54 because of 224x224 images
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.cnn_stack(x)
