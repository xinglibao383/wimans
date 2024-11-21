import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.3, backbone=None, interpolate=False):
        super(ResNet, self).__init__()

        self.interpolate = interpolate

        self.conv1 = nn.Conv2d(in_channels=270, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(3)

        self.dropout = nn.Dropout(p=dropout)

        self.resnet = backbone or models.resnet18(weights=None)
        self.embed_dim = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])

        self.project = nn.Linear(self.get_project_in_features(), hidden_dim)

    def forward(self, x):
        batch_size = x.size()[0]

        if self.interpolate:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.resnet(x)
        x = x.view(batch_size, -1)

        x = self.project(x)
        x = self.dropout(x)

        return x

    def get_project_in_features(self):
        x = torch.randn(1, 3, 224, 224)
        x = self.resnet(x)
        x = x.view(1, -1)
        return x.size()[1]


class Transformer(nn.Module):
    def __init__(self, input_dim=270, hidden_dim=1024, nhead=8, encoder_layers=6, dropout=0.3, with_positional=False):
        super(Transformer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        self.dropout = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv1d(in_channels=270, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=270, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(270)

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        # todo xinglibao: max_len should be computed, and is affected by conv1 and conv2
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=750) if with_positional else None
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=nhead),
                                             num_layers=encoder_layers)

    def forward(self, x):
        batch_size, time_steps, _, _, _ = x.shape
        # [batch_size, time_steps, transmitter, receiver, subcarrier] -> [batch_size, time_steps, transmitter * receiver * subcarrier]
        x = x.view(batch_size, time_steps, -1)
        # [batch_size, time_steps, transmitter * receiver * subcarrier] -> [batch_size, transmitter * receiver * subcarrier, time_steps]
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # [batch_size, transmitter * receiver * subcarrier, time_steps] -> [batch_size, time_steps, transmitter * receiver * subcarrier]
        x = x.permute(0, 2, 1)

        # [batch_size, time_steps, input_dim=270] -> [batch_size, time_steps, hidden_dim]
        x = self.input_linear(x)
        x = self.dropout(x)

        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        x = x.permute(1, 0, 2)
        x = self.encoder(x)

        return x.mean(dim=0)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, time_steps, hidden_dim]
        return x + self.pe[:, :x.size(1), :]


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=270, hidden_dim=1024, nhead=8, encoder_layers=6, dropout1=0.3, dropout2=0.3,
                 feature_extractor1_name='transformer', feature_extractor2_name='resnet',
                 transformer_with_positional=False):
        super(FeatureExtractor, self).__init__()

        if feature_extractor1_name == 'transformer':
            self.feature_extractor1 = Transformer(input_dim, hidden_dim, nhead, encoder_layers, dropout1,
                                                  transformer_with_positional)

        if feature_extractor2_name == 'resnet':
            self.feature_extractor2 = ResNet(hidden_dim, dropout2)

    def forward(self, x1, x2):
        return torch.cat((self.feature_extractor1(x1), self.feature_extractor2(x2)), dim=1)


class Classifier(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.3, num_users=6, num_locations=5, num_activities=9):
        super(Classifier, self).__init__()

        self.num_users = num_users
        self.num_locations = num_locations + 1
        self.num_activities = num_activities + 1

        self.hidden_dim = hidden_dim * 2

        self.head1 = nn.Linear(self.hidden_dim, self.num_users * 2)
        self.head2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, self.num_users * self.num_locations),
        )
        self.head3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 4, self.num_users * self.num_activities),
        )

    def forward(self, x):
        return self.head1(x), self.head2(x), self.head3(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.3):
        super(DomainDiscriminator, self).__init__()

        self.num_domain = 3
        self.hidden_dim = hidden_dim * 2 + self.num_domain
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, (self.hidden_dim - self.num_domain) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((self.hidden_dim - self.num_domain) // 2, (self.hidden_dim - self.num_domain) // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((self.hidden_dim - self.num_domain) // 4, self.num_domain),
        )

    def forward(self, x):
        return self.head(x)


if __name__ == "__main__":
    feature_extractor = FeatureExtractor(hidden_dim=1024, transformer_with_positional=True)
    classifier = Classifier(hidden_dim=1024)
    domain_discriminator = DomainDiscriminator(hidden_dim=1024)

    x1 = torch.randn(1, 600, 3, 3, 30)
    x2 = torch.randn(1, 270, 62, 9)

    x = feature_extractor(x1, x2)
    _, _, y1 = classifier(x)
    y2 = domain_discriminator(torch.cat((x, torch.from_numpy(np.array([[0, 1, 0]]))), dim=1))

    print("X1 shape:", x1.shape)
    print("X2 shape:", x2.shape)
    print("X shape:", x.shape)
    print("Y1 shape:", y1.shape)
    print("Y2 shape:", y2.shape)
