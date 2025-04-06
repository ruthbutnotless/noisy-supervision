import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Define transformations for CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Function to introduce label noise
def add_label_noise(dataset, noise_level=0.2):
    num_classes = 10
    num_samples = len(dataset.targets)
    noisy_targets = dataset.targets.copy()
    
    num_noisy = int(noise_level * num_samples)
    noisy_indices = np.random.choice(np.arange(num_samples), num_noisy, replace=False)
    
    for idx in noisy_indices:
        noisy_targets[idx] = np.random.choice([i for i in range(num_classes) if i != noisy_targets[idx]])

    dataset.targets = noisy_targets
    return dataset

# Create noisy dataset (20% label noise)
train_dataset_noisy = add_label_noise(train_dataset, noise_level=0.2)

# Define a simple feedforward neural network (FNN)
class SimpleFNN(nn.Module):
    def __init__(self):
        super(SimpleFNN, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)  # Flatten the input image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function
def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    accuracy_history = []
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs, labels  # Move to CPU (macOS M1 users should avoid CUDA issues)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct / total
        accuracy_history.append(epoch_accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    return accuracy_history

# Main execution block to prevent multiprocessing errors on macOS
if __name__ == '__main__':
    # Create DataLoaders (num_workers=0 for macOS compatibility)
    trainloader_clean = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    trainloader_noisy = DataLoader(train_dataset_noisy, batch_size=64, shuffle=True, num_workers=0)

    # Initialize model, loss function, and optimizer
    model = SimpleFNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train on clean dataset
    print("Training on clean labels...")
    clean_accuracy = train_model(model, trainloader_clean, criterion, optimizer, epochs=10)

    # Train on noisy dataset
    print("\nTraining on noisy labels...")
    noisy_accuracy = train_model(model, trainloader_noisy, criterion, optimizer, epochs=10)

    # Plot accuracy comparison
    epochs = np.arange(1, 11)
    plt.plot(epochs, clean_accuracy, label='Clean Labels')
    plt.plot(epochs, noisy_accuracy, label='Noisy Labels')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: Clean vs Noisy Labels')
    plt.legend()
    plt.show()
