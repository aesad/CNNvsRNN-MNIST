class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        # input is (batch_size, in_channels, H, W) with H = W = 28
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1) #(batch_size, 32, H, W)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # (batch_size, (32 or 64), H/2, W/2)
        # Flatten layer
        self.flatten = nn.Flatten() # (batch_size, 64 * (H/2) * (W/2))
        # d
        self.dropout = nn.Dropout(0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x