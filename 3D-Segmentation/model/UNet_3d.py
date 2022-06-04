# -*- coding = utf-8 -*-
# @Time : 2021/6/1 16:20
# @Author : Luxlios
# @File : UNet_3d.py
# @Software : PyCharm

import torch
import torch.nn as nn

class UNet_3d(nn.Module):
    def __init__(self, in_channel, n_class):
        '''
        :param in_channel: input's channel
        :param n_class: number of class
        '''
        super(UNet_3d, self).__init__()
        # input: in_channel * depth * 128 * 128
        # left
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(inplace=True),
            # 32 * depth * 128 * 128
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            # 64 * depth * 128 * 128
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            # 64 * depth / 2 * 64 * 64
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            # 64 * depth / 2 * 64 * 64
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True),
            # 128 * depth / 2 * 64 * 64
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            # 128 * depth / 4 * 32 * 32
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True),
            # 128 * depth / 4 * 32 * 32
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            # 256 * depth / 4 * 32 * 32
        )

        self.bottom = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            # 256 * depth / 8 * 16 * 16
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            # 256 * depth / 8 * 16 * 16
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
            # 512 * depth / 8 * 16 * 16
            nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, output_padding=0)
            # 512 * depth / 4 * 32 * 32
        )

        # right
        # cat -> 768 * depth / 4 * 32 * 32
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            # 256 * depth / 4 * 32 * 32
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            # 256 * depth / 4 * 32 * 32
            nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, output_padding=0)
            # 256 * depth / 2 * 64 * 64
        )
        # cat -> 384 * depth /2 * 64 * 64
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True),
            # 128 * depth / 2 * 64 * 64
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True),
            # 128 * depth / 2 * 64 * 64
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, output_padding=0)
            # 128 * depth * 128 * 128
        )
        # cat -> 192 * depth * 128 *128
        self.decoder3 = nn.Sequential(
            nn.Conv3d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            # 64 * depth * 128 * 128
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            # 64 * depth * 128 * 128
            nn.Conv3d(in_channels=64, out_channels=n_class, kernel_size=1, stride=1, padding=0)
            # n_class * depth * 128 *128
        )

    def forward(self, x):
        # left
        x = self.encoder1(x)
        x1 = x
        x = self.encoder2(x)
        x2 = x
        x = self.encoder3(x)
        x3 = x

        # bottom
        x = self.bottom(x)

        # decode
        x = torch.cat([x, x3], dim=1)
        x = self.decoder1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder3(x)
        return x

if __name__ == '__main__':
    x = torch.FloatTensor(1, 4, 32, 128, 128)  # batch * channel * d * h * w
    network = UNet_3d(in_channel=4, n_class=2)
    x = network(x)
    print(x.size())
