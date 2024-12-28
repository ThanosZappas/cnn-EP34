import torch
from torch import nn, optim

import utils
from classes.EarlyStopping import EarlyStopping

BATCH_SIZE = 64
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
    correct = 0
    total = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_function(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)

        if batch % 10 == 0:
            print(f"Batch {batch}: Loss = {loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Training Loss: {avg_loss:.6f}, Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy


def start_training_with_validation(model, train_dataloader, val_dataloader, optimizer, loss_function, device):
    early_stopping = EarlyStopping(patience=5, delta=0.5)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")

        # Training Phase
        train_loss, train_accuracy = train_one_epoch(model, train_dataloader, optimizer, loss_function, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation Phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_dataloader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Validation Loss: {val_loss:.6f}, Accuracy: {val_accuracy * 100:.2f}%")

        if early_stopping(train_loss, val_loss):
            print("Early stopping triggered.")
            break

    print("Done Training!")
    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    user_parameters = utils.get_user_input_for_training()
    BATCH_SIZE = user_parameters['batch_size']
    MAX_EPOCHS = user_parameters['max_epochs']
    dataset = user_parameters['dataset']
    train_ds, val_ds, test_ds = utils.train_val_test_split(dataset, [0.6, 0.2, 0.2])
    train_dataloader, val_dataloader, test_dataloader = utils.get_dataloaders(train_ds, val_ds, test_ds,
                                                                              BATCH_SIZE, shuffle=True)
    device = utils.get_device()
    print(f'Using device: {device}')

    model = utils.initialize_model(user_parameters['model_name'], device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3 if user_parameters['model_name'] in ["CNN1", "CNN2"] else 1e-4,
        betas=(0.9, 0.99)
    )

    print("\nStarting training...\n")
    start_training_with_validation(model, train_dataloader, val_dataloader, optimizer, loss_function, device)

    print("\nEvaluating on Test Set...")
    predict(model, test_dataloader, loss_function, device)
