import torch
import torch.nn as nn
import torchvision.models as models
from resize_model import ResizeModel


class ResNet(nn.Module):
    def __init__(self, backbone=None, num_users=6, num_classes=10):
        super(ResNet, self).__init__()

        self.num_users = num_users
        self.num_classes = num_classes

        self.resize_model = ResizeModel()

        self.resnet = backbone or models.resnet18(weights=None)
        self.embed_dim = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])

        self.head1 = nn.Linear(self.embed_dim, self.num_users * 2)
        self.head2 = nn.Linear(self.embed_dim, self.num_users * self.num_classes)

    def forward(self, x):
        x = self.resize_model(x)
        x = self.resnet(x)

        batch_size, _, _, _ = x.shape
        x = x.view(batch_size, self.embed_dim)

        y1, y2 = self.head1(x), self.head2(x)
        y1, y2 = y1.view(batch_size, self.num_users, 2), y2.view(batch_size, self.num_users, self.num_classes)

        return y1, y2


if __name__ == "__main__":
    batch_size, time, transmitter, receiver, subcarrier = 4, 3000, 3, 3, 30
    x = torch.randn(batch_size, time, transmitter, receiver, subcarrier)

    resnet = ResNet()

    y1, y2 = resnet(x)

    print("X shape:", x.shape)
    print("Y1 shape:", y1.shape)
    print("Y1 shape:", y2.shape)
