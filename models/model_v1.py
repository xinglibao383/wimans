import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.3, backbone=None):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=270, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1)

        self.dropout = nn.Dropout(p=dropout)

        self.resnet = backbone or models.resnet18(weights=None)
        self.embed_dim = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])

        self.project = nn.Linear(self.get_project_in_features(), hidden_dim)

    def forward(self, x):
        batch_size = x.size()[0]

        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.dropout(F.relu(self.conv4(x)))
        x = self.dropout(F.relu(self.conv5(x)))

        x = self.resnet(x)
        x = x.view(batch_size, -1)

        return self.project(x)

    def get_project_in_features(self):
        x = torch.randn(1, 3, 224, 224)
        x = self.resnet(x)
        x = x.view(1, -1)
        return x.size()[1]


class Transformer(nn.Module):
    def __init__(self, input_dim=270, hidden_dim=1024, nhead=8, encoder_layers=6, dropout=0.3):
        super(Transformer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        self.dropout = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv1d(in_channels=270, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=270, kernel_size=3, stride=3)

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=nhead),
                                             num_layers=encoder_layers)

    def forward(self, x):
        batch_size, time_steps, _, _, _ = x.shape
        # [batch_size, time_steps, transmitter, receiver, subcarrier] -> [batch_size, time_steps, transmitter * receiver * subcarrier]
        x = x.view(batch_size, time_steps, -1)
        # [batch_size, time_steps, transmitter * receiver * subcarrier] -> [batch_size, transmitter * receiver * subcarrier, time_steps]
        x = x.permute(0, 2, 1)

        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))

        # [batch_size, transmitter * receiver * subcarrier, time_steps] -> [batch_size, time_steps, transmitter * receiver * subcarrier]
        x = x.permute(0, 2, 1)

        # [batch_size, time_steps, input_dim=270] -> [batch_size, time_steps, hidden_dim]
        x = self.input_linear(x)

        x = x.permute(1, 0, 2)
        x = self.encoder(x)

        return x[-1, :, :]


class FeatureExtractorV1(nn.Module):
    def __init__(self, input_dim=270, hidden_dim=1024, nhead=8, encoder_layers=6, dropout1=0.3, dropout2=0.3):
        super(FeatureExtractorV1, self).__init__()

        self.transformer = Transformer(input_dim, hidden_dim, nhead, encoder_layers, dropout1)
        self.resnet = ResNet(hidden_dim, dropout2)

    def forward(self, x1, x2):
        return torch.cat((self.transformer(x1), self.resnet(x2)), dim=1)


class MyModel(nn.Module):
    def __init__(self, input_dim=270, hidden_dim=1024, nhead=8, encoder_layers=6, dropout1=0.3, dropout2=0.3,
                 num_users=6, num_locations=5, num_activities=9):
        super(MyModel, self).__init__()

        self.num_users = num_users
        self.num_locations = num_locations + 1
        self.num_activities = num_activities + 1

        self.hidden_dim = hidden_dim * 2

        self.feature_extractor = FeatureExtractorV1(input_dim, hidden_dim, nhead, encoder_layers, dropout1, dropout2)

        self.head1 = nn.Linear(self.hidden_dim, self.num_users * 2)
        self.head2 = nn.Linear(self.hidden_dim, self.num_users * self.num_locations)
        self.head3 = nn.Linear(self.hidden_dim, self.num_users * self.num_activities)

    def forward(self, x1, x2):
        x = self.feature_extractor(x1, x2)
        return self.head1(x), self.head2(x), self.head3(x)


if __name__ == "__main__":
    model = MyModel(hidden_dim=1024)

    x1 = torch.randn(32, 60, 3, 3, 30)
    x2 = torch.randn(32, 270, 123, 5)

    y1, y2, y3 = model(x1, x2)

    print("X1 shape:", x1.shape)
    print("X2 shape:", x2.shape)
    print("Y1 shape:", y1.shape)
    print("Y2 shape:", y2.shape)
    print("Y3 shape:", y3.shape)
