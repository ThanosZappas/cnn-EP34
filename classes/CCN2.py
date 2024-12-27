import torch.nn as nn


class CNN2(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2, self).__init__()
        self.cnn_stack = nn.Sequential(
            # First two convolutional layers with 32 filters (3x3) and ReLU
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max pooling with step 4
            nn.MaxPool2d(kernel_size=4, stride=4),

            # Next two convolutional layers with 64 filters (3x3) and ReLU
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max pooling with step 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Next two convolutional layers with 128 filters (3x3) and ReLU
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max pooling with step 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Three convolutional layers with 256 filters (3x3) and ReLU
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max pooling with step 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # One convolutional layer with 512 filters (3x3) and ReLU
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max pooling with step 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Flatten layer to convert the 3D output into a 1D tensor
            nn.Flatten(),

            # Fully connected layer with 1024 neurons and ReLU activation
            nn.Linear(512 * 3 * 3, 1024),  # Adjusted based on final output size (3x3)
            nn.ReLU(),

            # Output layer with num_classes outputs and softmax activation
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)  # Softmax to output probabilities for classification
        )

    def forward(self, x):
        return self.cnn_stack(x)
