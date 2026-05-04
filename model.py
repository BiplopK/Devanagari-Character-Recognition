import torch
import torch.nn as nn
import torch.nn.functional as F

class DevanagariNetwork(nn.Module):
    def __init__(self):
        super(DevanagariNetwork,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5,padding=2)  # Conv2d: 1 input channel, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling with 2x2 kernel
        self.conv2 = nn.Conv2d(32, 64, 5,padding=2)
        self.conv3=nn.Conv2d(64,128,5,padding=2)  # Conv2d: 6 input channels, 16 output channels, 5x5 kernel
        self.dropout=nn.Dropout(0.25)
        # Calculate the flattened dimension after conv and pooling layers
        self.fc1 = nn.Linear(128*3*3, 256)  # Adjusted flattened size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 58)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # After conv1 + pool, the size becomes 24x24
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x= self.dropout(x) # After conv2 + pool, the size becomes 4x4
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
