import torch.nn as nn
import torch.nn.functional as F


class ResizeModel(nn.Module):
    def __init__(self):
        super(ResizeModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=270, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=270, kernel_size=3, stride=3)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, time, _, _, _ = x.shape

        # [batch_size, time, transmitter, receiver, subcarrier] -> [batch_size, time, transmitter * receiver * subcarrier]
        x = x.view(batch_size, time, -1)
        # [batch_size, time, transmitter * receiver * subcarrier] -> [batch_size, transmitter * receiver * subcarrier, time]
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # [batch_size, transmitter * receiver * subcarrier, time] -> [batch_size, 1, transmitter * receiver * subcarrier, time]
        x = x.unsqueeze(1)

        # [batch_size, 1, transmitter * receiver * subcarrier, time] -> [batch_size, 3, transmitter * receiver * subcarrier, time]
        x = F.relu(self.conv4(x))

        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

        return x
