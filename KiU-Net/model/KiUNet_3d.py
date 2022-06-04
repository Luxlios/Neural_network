# -*- coding = utf-8 -*-
# @Time : 2021/6/1 15:30
# @Author : Luxlios
# @File : KiUNet.py
# @Software : PyCharm

import torch
import torch.nn as nn

class KiUNet_3d(nn.Module):
    def __init__(self, in_channel, n_base=1, n_class=2):
        '''
        :param in_channel: input's channel
        :param n_base: U-Net & Ki-Net 's base channel
        :param n_class: number of class
        '''
        super(KiUNet_3d, self).__init__()
        # in_channel * depth * height * width

        # Ki-Net
        # encoder
        self.encoder1_ki = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            # nn.BatchNorm3d(num_features=n_base),
            nn.ReLU(inplace=True)
        )
        # 16 * 2depth * 2height * 2width
        self.crfb1_u2ki = nn.Sequential(
            nn.Conv3d(in_channels=n_base, out_channels=n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4, mode='trilinear'),
            # nn.BatchNorm3d(num_features=n_base),
            nn.ReLU(inplace=True)
        )
        self.encoder2_ki = nn.Sequential(
            nn.Conv3d(in_channels=n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            # nn.BatchNorm3d(num_features=2*n_base),
            nn.ReLU(inplace=True)
        )
        # 32 * 4depth * 4height * 4width
        self.crfb2_u2ki = nn.Sequential(
            nn.Conv3d(in_channels=2*n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=16, mode='trilinear'),
            # nn.BatchNorm3d(num_features=2*n_base),
            nn.ReLU(inplace=True)
        )
        self.encoder3_ki = nn.Sequential(
            # nn.Conv3d(in_channels=2*n_base, out_channels=4*n_base, kernel_size=3, stride=1, padding=1),
            # kite net has too many parameters due to upsampling, so the channel should not rise too much
            nn.Conv3d(in_channels=2*n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            # nn.BatchNorm3d(num_features=4*n_base),
            nn.ReLU(inplace=True)
        )
        # 64 * 8depth * 8height * 8width
        self.crfb3_u2ki = nn.Sequential(
            # nn.Conv3d(in_channels=4*n_base, out_channels=4*n_base, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=4*n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=64, mode='trilinear'),
            # nn.BatchNorm3d(num_features=4*n_base),
            nn.ReLU(inplace=True)
        )
        # decoder
        self.decoder1_ki = nn.Sequential(
            # nn.Conv3d(in_channels=4*n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=2*n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.BatchNorm3d(num_features=2*n_base),
            nn.ReLU(inplace=True)
        )
        # 32 * 4depth * 4height * 4width
        self.crfb4_u2ki = nn.Sequential(
            nn.Conv3d(in_channels=2*n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=16, mode='trilinear'),
            # nn.BatchNorm3d(num_features=2*n_base),
            nn.ReLU(inplace=True)
        )
        self.decoder2_ki = nn.Sequential(
            nn.Conv3d(in_channels=2*n_base, out_channels=n_base, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.BatchNorm3d(num_features=n_base),
            nn.ReLU(inplace=True)
        )
        # 16 * 2depth * 2height * 2width
        self.crfb5_u2ki = nn.Sequential(
            nn.Conv3d(in_channels=n_base, out_channels=n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4, mode='trilinear'),
            # nn.BatchNorm3d(num_features=n_base),
            nn.ReLU(inplace=True)
        )
        self.decoder3_ki = nn.Sequential(
            nn.Conv3d(in_channels=n_base, out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.BatchNorm3d(num_features=in_channel),
            nn.ReLU(inplace=True)
        )
        # 8 * depth * height * width

        # U-Net
        # encoder
        self.encoder1_u = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=n_base, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.BatchNorm3d(num_features=n_base),
            nn.ReLU(inplace=True)
        )
        # 16 * depth/2 * height/2 * width/2
        self.crfb1_ki2u = nn.Sequential(
            nn.Conv3d(in_channels=n_base, out_channels=n_base, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool3d(kernel_size=4, stride=4),
            nn.Upsample(scale_factor=0.25, mode='trilinear'),
            # nn.BatchNorm3d(num_features=n_base),
            nn.ReLU(inplace=True)
        )
        self.encoder2_u = nn.Sequential(
            nn.Conv3d(in_channels=n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.BatchNorm3d(num_features=2*n_base),
            nn.ReLU(inplace=True)
        )
        # 32 * depth/4 * height/4 * width/4
        self.crfb2_ki2u = nn.Sequential(
            nn.Conv3d(in_channels=2*n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool3d(kernel_size=16, stride=16),
            nn.Upsample(scale_factor=0.0625, mode='trilinear'),
            # nn.BatchNorm3d(num_features=2*n_base),
            nn.ReLU(inplace=True)
        )
        self.encoder3_u = nn.Sequential(
            nn.Conv3d(in_channels=2*n_base, out_channels=4*n_base, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.BatchNorm3d(num_features=4*n_base),
            nn.ReLU(inplace=True)
        )
        # 64 * depth/8 * height/8 * width/8
        self.crfb3_ki2u = nn.Sequential(
            # nn.Conv3d(in_channels=4*n_base, out_channels=4*n_base, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=2*n_base, out_channels=4*n_base, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool3d(kernel_size=64, stride=64),
            nn.Upsample(scale_factor=0.015625, mode='trilinear'),
            # nn.BatchNorm3d(num_features=4*n_base),
            nn.ReLU(inplace=True)
        )
        # decoder
        self.decoder1_u = nn.Sequential(
            nn.Conv3d(in_channels=4*n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            # nn.BatchNorm3d(num_features=2*n_base),
            nn.ReLU(inplace=True)
        )
        # 32 * depth/4 * height/4 * width/4
        self.crfb4_ki2u = nn.Sequential(
            nn.Conv3d(in_channels=2*n_base, out_channels=2*n_base, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool3d(kernel_size=16, stride=16),
            nn.Upsample(scale_factor=0.0625, mode='trilinear'),
            # nn.BatchNorm3d(num_features=2*n_base),
            nn.ReLU(inplace=True)
        )
        self.decoder2_u = nn.Sequential(
            nn.Conv3d(in_channels=2*n_base, out_channels=n_base, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            # nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=True)
        )
        # 16 * depth/2 * height/2 * width/2
        self.crfb5_ki2u = nn.Sequential(
            nn.Conv3d(in_channels=n_base, out_channels=n_base, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool3d(kernel_size=4, stride=4),
            nn.Upsample(scale_factor=0.25, mode='trilinear'),
            # nn.BatchNorm3d(num_features=n_base),
            nn.ReLU(inplace=True)
        )
        self.decoder3_u = nn.Sequential(
            nn.Conv3d(in_channels=n_base, out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            # nn.BatchNorm3d(num_features=8),
            nn.ReLU(inplace=True)
        )
        # 8 * depth * height * width

        # last
        self.classifier = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=n_class, kernel_size=1, stride=1, padding=0)
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # layer 1
        x1, x2 = self.encoder1_ki(x), self.encoder1_u(x)
        k1, u1 = x1, x2

        # layer 2
        x1 = torch.add(x1, self.crfb1_u2ki(u1))
        x1 = self.encoder2_ki(x1)
        k2 = x1
        x2 = torch.add(x2, self.crfb1_ki2u(k1))
        x2 = self.encoder2_u(x2)
        u2 = x2

        # layer 3
        x1 = torch.add(x1, self.crfb2_u2ki(u2))
        x1 = self.encoder3_ki(x1)
        k3 = x1
        x2 = torch.add(x2, self.crfb2_ki2u(k2))
        x2 = self.encoder3_u(x2)
        u3 = x2

        # layer 4
        x1 = torch.add(x1, self.crfb3_u2ki(u3))
        x1 = self.decoder1_ki(x1)
        k4 = x1
        x2 = torch.add(x2, self.crfb3_ki2u(k3))
        x2 = self.decoder1_u(x2)
        u4 = x2

        # layer 5
        x1 = torch.add(x1, self.crfb4_u2ki(u4))
        x1 = torch.add(x1, k2)  # skip connection
        x1 = self.decoder2_ki(x1)
        k5 = x1
        x2 = torch.add(x2, self.crfb4_ki2u(k4))
        x2 = torch.add(x2, u2)  # skip connection
        x2 = self.decoder2_u(x2)
        u5 = x2

        # layer 6
        x1 = torch.add(x1, self.crfb5_u2ki(u5))
        x1 = torch.add(x1, k1)  # skip connection
        x1 = self.decoder3_ki(x1)
        # k6 = x1
        x2 = torch.add(x2, self.crfb5_ki2u(k5))
        x2 = torch.add(x2, u1)  # skip connection
        x2 = self.decoder3_u(x2)
        # u6 = x2

        # last
        x = torch.add(x1, x2)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    x = torch.FloatTensor(1, 4, 32, 128, 128)  # batch * channel * depth * height * width
    network = KiUNet_3d(in_channel=4, n_base=1, n_class=2)
    x = network(x)
    print(x.size())


