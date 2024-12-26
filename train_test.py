import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader, random_split
from COVID19Dataset import get_dataset
from cnn1 import CNN1

DATA_ROOT_DIR = "datasets/Testing_COVID-19_Radiography_Dataset"
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
BATCH_SIZE = 64
MAX_EPOCHS = 20

def get_confusion_matrix(y, y_pred, num_classes):
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true_class, pred_class in zip(y, y_pred):
        confusion_matrix[true_class, pred_class] += 1
    return confusion_matrix


def display_confusion_matrix(confusion_matrix, num_classes):
    confusion_matrix_np = confusion_matrix.numpy()
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_np,
                                                      display_labels=range(num_classes))
    confusion_matrix_display.plot()
    plt.show()
    print("Confusion Matrix: \n", confusion_matrix.numpy())


def get_class_name(one_hot_encoded_label):
    return CLASSES[one_hot_encoded_label.argmax().item()]


#PLAYING WITH THE DATA
def display_random_image(dataloader):
    # Display image and label.
    inputs, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {inputs.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = inputs[0].squeeze()
    label = train_labels[0]
    img = img.swapaxes(0, 1)
    img = img.swapaxes(1, 2)
    plt.title(get_class_name(label))
    plt.imshow(img, cmap="gray")
    plt.show()
    # print(CLASSES[label])


def train_val_test_split(dataset):
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [0.6, 0.2, 0.2], generator=generator)


def train_one_epoch(model, dataloader, optimizer, loss_function, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_function(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def predict(model, dataloader, loss_function, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, matrix = 0, 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            matrix += get_confusion_matrix(y, pred.argmax(1), len(CLASSES))

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    display_confusion_matrix(matrix, len(CLASSES))

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def start_training(model, train_dataloader, optimizer, loss_function, device):
    for t in range(MAX_EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_one_epoch(model, train_dataloader, optimizer, loss_function, device)
        # predict(model, test_dataloader, loss_function, device)
    print("Done Training!")


if __name__ == '__main__':
    train_ds, value_ds, test_ds = train_val_test_split(get_dataset(DATA_ROOT_DIR))
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    value_dataloader = DataLoader(value_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    device = get_device()
    print(f'Using device: {device}')

    model = CNN1()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    start_training(model, train_dataloader, optimizer, loss_function, device)
    predict(model, test_dataloader, loss_function, device)