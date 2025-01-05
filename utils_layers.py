import torch.nn as nn


class AtariDQNConv(nn.Module):
    def __init__(self, **kwargs):
        super(AtariDQNConv, self).__init__()
        self.conv = nn.Conv2d(**kwargs)

        nn.init.orthogonal_(self.conv.weight, gain=nn.init.calculate_gain(nonlinearity='relu'))

    def forward(self, input):
        return self.conv(input)


class AtariDQNNetwork(nn.Module):
    def __init__(self, num_actions):
        super(AtariDQNNetwork, self).__init__()

        self.network = nn.Sequential(
            AtariDQNConv(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            AtariDQNConv(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            AtariDQNConv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

    def forward(self, input):
        return self.network(input / 255.0)
