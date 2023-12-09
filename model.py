import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, vocab_size):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv3d(1, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = nn.Conv3d(256, 75, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.flatten = nn.Flatten()
        self.time_dist = nn.Sequential(nn.Linear(75 * 5 * 17, 128), nn.ReLU())

        self.lstm1 = nn.LSTM(128, 128, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(256, 128, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)

        self.dense = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # x = self.flatten(x)
        size = x.size()
        x = x.view(size[0], size[1], -1)
        x = self.time_dist(x)

        x, _ = self.lstm1(x.permute(1,0,2))

        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = self.dense(x)

        return x