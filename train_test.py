import torch
import torchvision
from torch import nn, optim

import utils
from classes.CCN2 import CNN2
from classes.EarlyStopping import EarlyStopping
from classes.CNN1 import CNN1

BATCH_SIZE = 32
MAX_EPOCHS = 5
CLASSES = utils.CLASSES
DATA_ROOT_DIR = utils.DATA_ROOT_DIR


def predict(model, dataloader, loss_function, device):
    model.eval()  # Set the model to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    matrix = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)  # Initialize confusion matrix

    with torch.no_grad():  # No gradient computation during evaluation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            matrix += utils.get_confusion_matrix(y, pred.argmax(1), len(CLASSES))

    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    utils.display_confusion_matrix(matrix)


def train_one_epoch(model, dataloader, optimizer, loss_function, device):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_function(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 10 == 0:
            print(f"Batch {batch}: Loss = {loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.6f}")
    return avg_loss


def start_training_with_validation(model, train_dataloader, val_dataloader, optimizer, loss_function, device):
    early_stopping = EarlyStopping(patience=5, delta=0.5)
    train_losses, val_losses = [], []

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")

        # Training Phase
        train_loss = train_one_epoch(model, train_dataloader, optimizer, loss_function, device)
        train_losses.append(train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_function(pred, y).item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        print(f"Validation Loss: {val_loss:.6f}")

        # Early Stopping Check
        if early_stopping(train_loss, val_loss):
            print("Early stopping triggered.")
            break

    print("Done Training!")


if __name__ == '__main__':
    train_ds, val_ds, test_ds = utils.train_val_test_split(utils.get_dataset(DATA_ROOT_DIR), [0.6, 0.2, 0.2])
    train_dataloader, val_dataloader, test_dataloader = utils.get_dataloaders(train_ds, val_ds, test_ds, BATCH_SIZE,
                                                                              shuffle=True)
    device = utils.get_device()
    print(f'Using device: {device}')

    cnn1 = CNN1()
    cnn2 = CNN2()
    pretrained_resnet50 = torchvision.models.resnet50(pretrained=True)

    model = pretrained_resnet50
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))  # learning rate: cnn: 1e-3, resnet50: 1e-4

    start_training_with_validation(model, train_dataloader, val_dataloader, optimizer, loss_function, device)

    print("\nEvaluating on Test Set:\n-------------------------------")
    predict(model, test_dataloader, loss_function, device)
