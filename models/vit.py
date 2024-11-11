import torch
import torch.nn as nn
from torchvision import models
from resize_model import ResizeModel


class VisionTransformer(torch.nn.Module):
    def __init__(self, backbone=None, num_users=6, num_classes=10):
        super(VisionTransformer, self).__init__()

        self.num_users = num_users
        self.num_classes = num_classes

        self.resize_model = ResizeModel()

        self.vit = backbone or models.vit_b_16(weights=None)
        self.vit.heads = torch.nn.Identity()
        self.embed_dim = self.vit.conv_proj.out_channels

        self.head1 = nn.Linear(self.embed_dim, self.num_users * 2)
        self.head2 = nn.Linear(self.embed_dim, self.num_users * self.num_classes)

    def forward(self, x):
        x = self.resize_model(x)
        batch_size, _, _, _ = x.shape

        x = self.vit(x)

        y1, y2 = self.head1(x), self.head2(x)
        y1, y2 = y1.view(batch_size, self.num_users, 2), y2.view(batch_size, self.num_users, self.num_classes)

        return y1, y2


if __name__ == "__main__":
    batch_size, time, transmitter, receiver, subcarrier = 4, 3000, 3, 3, 30
    x = torch.randn(batch_size, time, transmitter, receiver, subcarrier)

    vit = VisionTransformer()

    y1, y2 = vit(x)

    print("X shape:", x.shape)
    print("Y1 shape:", y1.shape)
    print("Y1 shape:", y2.shape)
