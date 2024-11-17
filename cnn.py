import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Conv1d(
                in_channels=2, out_channels=64, kernel_size=32,
                stride=1
                )

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=16, stride=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=8, stride=1
        )
        self.conv4 = nn.Conv1d(
            in_channels=256, out_channels=512, kernel_size=4, stride=1
        )
        self.conv5 = nn.Conv1d(
            in_channels=512, out_channels=768, kernel_size=2, stride=1
        )
        self.dropout = nn.Dropout1d(0.2)
        self.bn = nn.BatchNorm1d(num_features=768)

        self.ffn = nn.Sequential(
            nn.Flatten(),
            nn.AvgPool1d(kernel_size=3),
            nn.Linear(10752, 1024),
            nn.ReLU(),
            nn.Linear(1024, 5),
        )

    def forward(self, x):
        # x shape -> (B, num_features, num_points)
        # x = x.unsqueeze(1)
        # x shape -> (B, 1, num_features, num_points)
        out = self.relu(self.conv(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.relu(self.conv5(out))
        out = self.bn(self.dropout(out))
        out = self.ffn(out)
        out = out.squeeze(1)
        return out
