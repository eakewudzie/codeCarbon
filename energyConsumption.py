import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
import time
import os

# 🔹 Step 1: Load MNIST Dataset from Local Folder
# DATA_PATH = 'C:/Users/EUNICE/Desktop/miniproject/data'  # Ensure this matches your actual path
DATA_PATH = './data'  # Relative to the script location

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Ensure dataset exists
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Dataset not found in {DATA_PATH}. Ensure the MNIST files exist.")

# Load dataset from local folder
train_dataset = datasets.MNIST(root=DATA_PATH, train=True, transform=transform, download=False)
# train_dataset = datasets.MNIST(root=DATA_PATH, train=True, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 🔹 Step 2: Define a Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 🔹 Step 3: Train the Model and Measure Energy Consumption
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

tracker = EmissionsTracker()  # Start energy tracking
tracker.start()

start_time = time.time()
epochs = 10  # You can adjust this

for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

tracker.stop()  # Stop tracking
total_time = time.time() - start_time

# 🔹 Step 4: Log Energy Consumption
print(f"Total Training Time: {total_time:.2f} seconds")
print(f"Energy Consumed (kgCO2): {tracker.final_emissions}")
