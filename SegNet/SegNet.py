# -*- coding = utf-8 -*-
# @Time : 2022/5/3 16:32
# @Author : Luxlios
# @File : SegNet.py
# @Software : PyCharm

import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        # 64 * 360 * 480
        # 64 * 180 * 240
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        # 128 * 180 * 240
        # 128 * 90 * 120
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        # 256 * 90 * 120
        # 256 * 45 * 60
        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )
        # 512 * 45 * 60
        # 512 * 23 * 30(nn.MaxPool2d(ceil_mode=True))
        self.encoder5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )
        # 512 * 23 * 30
        # 512 * 12 * 15(nn.MaxPool2d(ceil_mode=True))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)
    def forward(self, x):
        indice = []
        x = self.encoder1(x)
        x, indice_temp = self.pool(x)
        indice.append(indice_temp)
        x = self.encoder2(x)
        x, indice_temp = self.pool(x)
        indice.append(indice_temp)
        x = self.encoder3(x)
        x, indice_temp = self.pool(x)
        indice.append(indice_temp)
        x = self.encoder4(x)
        x, indice_temp = self.pool(x)
        indice.append(indice_temp)
        x = self.encoder5(x)
        x, indice_temp = self.pool(x)
        indice.append(indice_temp)
        return x, indice

class segnet(nn.Module):
    def __init__(self, n_class, vgg16_rmfc):
        super(segnet, self).__init__()
        # input: 3 * 360 * 480
        # VGG first 5 layers
        self.encoder = vgg16_rmfc
        # 512 * 12 * 15(nn.MaxPool2d(ceil_mode=True))

        # 512 * 23 * 30
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )
        # 512 * 23 * 30
        # 512 * 45 * 60
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        # 256 * 45 * 60
        # 256 * 90 * 120
        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        # 128 * 90 * 120
        # 128 * 180 * 240
        self.decoder4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        # 64 * 180 * 240
        # 64 * 360 * 480
        self.decoder5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )
        # 3 * 360 * 480
        self.depool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # encoder
        x, indice = self.encoder(x)
        # decoder
        x = self.depool(x, indice[4])
        x = x[:, :, :-1, :]  # one more column for ceil_mode=True
        x = self.decoder1(x)
        x = self.depool(x, indice[3])
        x = x[:, :, :-1, :]  # one more column for ceil_mode=True
        x = self.decoder2(x)
        x = self.depool(x, indice[2])
        x = self.decoder3(x)
        x = self.depool(x, indice[1])
        x = self.decoder4(x)
        x = self.depool(x, indice[0])
        x = self.decoder5(x)
        return x

if __name__ == '__main__':
    x = torch.FloatTensor(1, 3, 360, 480)
    vgg16_rmfc = encoder()
    network = segnet(n_class=11, vgg16_rmfc=vgg16_rmfc)
    x = network(x)
    print(x.size())



