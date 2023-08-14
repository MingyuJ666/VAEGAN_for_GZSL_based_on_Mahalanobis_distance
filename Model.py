import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

n_c = 32
# Encoder network
# Need trans h_dim, z_dim, image_size
class Encoder(nn.Module):
    def __init__(self, h_dim, z_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = nn.Linear(512 * 16 * 16, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        h = self.fc1(x)
        z = self.fc2(h)
        return z, h


# Decoder network
class Decoder(nn.Module):
    def __init__(self, h_dim, z_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 512 * 16 * 16)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc1(z)
        x = self.fc2(h)
        x = x.view(x.size(0), 512, 16, 16)
        x = self.conv(x)
        return x


# Discriminator network
class DiscriminatorA(nn.Module):
    def __init__(self, image_size):
        super(DiscriminatorA, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.image_size = image_size

    def forward(self, x):
        x = self.conv(x)
        return x.view(-1, 1)



class ResidualBlockD(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.scale_conv = None
        if in_channels != out_channels:
            self.scale_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.scale_conv is not None:
            x = self.scale_conv(x)

        return F.avg_pool2d(x, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self._shortcut(x) + self.gamma * self.residual_conv(x)

class DiscriminatorB(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        self.img_forward = nn.Sequential(
            # [batch_size, 3, h, w]
            nn.Conv2d(3, n_c*2, kernel_size=3, stride=2, padding=1),
            ResidualBlockD(n_c * 2, n_c * 4),
            ResidualBlockD(n_c * 4, n_c * 8),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.img_forward(x)
        return out.view(-1, 1)

