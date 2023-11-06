import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPair(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(ConvPair, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        self.down_convb = nn.Sequential(
            nn.MaxPool2d(2),
            ConvPair(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        x = self.down_convb(x)
        return x


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels): #, bilinear=True):
        super(UpSamplingBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.convp = ConvPair(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, xi):
        x = self.up(x)

        diffY = xi.size()[2] - x.size()[2]
        diffX = xi.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, xi], dim=1)
        x = self.convp(x)

        return x

class OutConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvBlock, self).__init__()
        self.out_convb = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels//2),
            nn.Tanh(),
            nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.out_convb(x)

        return x

