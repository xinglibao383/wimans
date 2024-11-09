import torch
import torch.nn as nn


class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(30, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, time, transmitter, receiver, subcarrier = x.size()
        x = x.view(batch_size * time, transmitter, receiver, subcarrier)
        # [batch_size * time, transmitter, receiver, subcarrier] -> [batch_size * time, subcarrier, transmitter, receiver]
        x = x.permute(0, 3, 1, 2)

        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.relu(self.deconv5(x))
        x = self.deconv6(x)

        # [batch_size * time, subcarrier, transmitter, receiver] -> [batch_size, time, subcarrier, transmitter, receiver]
        x = x.view(batch_size, time, 3, 192, 192)
        # [batch_size, time, subcarrier, transmitter, receiver] -> [batch_size, subcarrier, time, transmitter, receiver]
        x = x.permute(0, 2, 1, 3, 4)
        return x


if __name__ == "__main__":
    # [batch_size, time, transmitter, receiver, subcarrier]
    x = torch.randn(2, 20, 3, 3, 30)
    model = DeconvNet()
    # [batch_size, subcarrier, time, transmitter, receiver]
    y = model(x)
    print(y.shape)