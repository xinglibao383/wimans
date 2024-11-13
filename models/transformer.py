import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, input_dim=270, hidden_dim=512, nhead=8, encoder_layers=6,
                 num_users=6, num_locations=5, num_activities=9):
        super(Transformer, self).__init__()

        self.num_users = num_users
        self.num_locations = num_locations + 1
        self.num_activities = num_activities + 1

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        self.conv1 = nn.Conv1d(in_channels=270, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=270, kernel_size=3, stride=3)

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=nhead),
                                             num_layers=encoder_layers)

        self.head1 = nn.Linear(hidden_dim, self.num_users * 2)
        self.head2 = nn.Linear(hidden_dim, self.num_users * self.num_locations)
        self.head3 = nn.Linear(hidden_dim, self.num_users * self.num_activities)

    def forward(self, x):
        batch_size, time_steps, _, _, _ = x.shape
        # [batch_size, time_steps, transmitter, receiver, subcarrier] -> [batch_size, time_steps, transmitter * receiver * subcarrier]
        x = x.view(batch_size, time_steps, -1)
        # [batch_size, time_steps, transmitter * receiver * subcarrier] -> [batch_size, transmitter * receiver * subcarrier, time_steps]
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # [batch_size, transmitter * receiver * subcarrier, time_steps] -> [batch_size, time_steps, transmitter * receiver * subcarrier]
        x = x.permute(0, 2, 1)

        # [batch_size, time_steps, input_dim=270] -> [batch_size, time_steps, hidden_dim]
        x = self.input_linear(x)

        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x[-1, :, :]

        y1 = self.head1(x)
        y2 = self.head2(x)
        y3 = self.head3(x)

        return y1, y2, y3


if __name__ == "__main__":
    batch_size, time_steps, transmitter, receiver, subcarrier = 4, 50, 3, 3, 30
    x = torch.randn(batch_size, time_steps, transmitter, receiver, subcarrier)

    model = Transformer()
    y1, y2, y3 = model(x)
    print(y1.shape, y2.shape, y3.shape)
