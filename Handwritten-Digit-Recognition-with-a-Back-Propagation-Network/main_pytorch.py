import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Define a transformation to convert the data to a Pytorch tensor and normalize it.
# Normalize the pixel value to (-1 to 1). Images are grayscale so we normalize single channel.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network architecture
class DigitRecognitionCNN(nn.Module):
    def __init__(self):
        super(DigitRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # 1 input channel (grayscale), 6 output channels, 5x5 filter
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 6 input channels from conv1, 16 output channels
        
        #Fully Connected Layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # after max pooling, image size becomes 4x4, we need to flatten
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 output classes (digits 0-9)
        
    def forward(self, x):
        # Convolution -> ReLU -> Pooling (LeCun used tanh, but ReLU is commonly used today)
        x = F.relu(self.conv1(x)) # First convolution and activation
        x = F.max_pool2d(x, 2) # Max pooling (downsampling by 2)
        x = F.relu(self.conv2(x)) # Second convolution and activation
        x = F.max_pool2d(x, 2) # Max pooling (downsampling by 2)
        
        x = x.view(-1, 16 * 4 * 4) # flatten the image into fully connected layers
        x = F.relu(self.fc1(x)) # First fully connected layer
        x = F.relu(self.fc2(x)) # Second fully connected layer
        x = self.fc3(x) # Output layer (no activation needed here)
        
        return x


# Training the model
model = DigitRecognitionCNN() # Initialize the model
    
# Define the loss function and oprimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training Loop
for epoch in range(10):
    running_loss = 0.0
    for images, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward Pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print loss for every 100 mini-batches
        running_loss += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/10], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
            
            
# Test the model
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation for testing
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
