# -*- coding = utf-8 -*-
# @Time : 2022/4/22 17:12
# @Author : Luxlios
# @File : ResNet_50.py
# @Software : PyCharm

import torch
import torch.nn as nn

class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(basic_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.project = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        fx = self.f(x)
        if self.in_channels == self.out_channels:
            x = torch.add(x, fx)
            x = self.bn_relu(x)
        else:
            x1 = self.project(x)
            x = torch.add(x1, fx)
            x = self.bn_relu(x)
        return x

class bottleneck_block(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels, stride=1):
        super(bottleneck_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(num_features=out_channels1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels1, out_channels=out_channels1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels1, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.project = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        fx = self.f(x)
        if self.in_channels == self.out_channels:
            x = torch.add(x, fx)
            x = self.bn_relu(x)
        else:
            x1 = self.project(x)
            x = torch.add(x1, fx)
            x = self.bn_relu(x)
        return x

class resnet_50(nn.Module):
    def __init__(self, n_class=1000):
        super(resnet_50, self).__init__()
        # input: 3 * 224 * 224

        # convolution & pooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        # 64 * 112 * 112
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 64 * 56 * 56
            bottleneck_block(in_channels=64, out_channels1=64, out_channels=256, stride=1),
            # 256 * 56 * 56
            bottleneck_block(in_channels=256, out_channels1=64, out_channels=256, stride=1),
            # 256 * 56 * 56
            bottleneck_block(in_channels=256, out_channels1=64, out_channels=256, stride=1)
            # 256 * 56 * 56
        )
        # 256 * 56 * 56
        self.conv3 = nn.Sequential(
            bottleneck_block(in_channels=256, out_channels1=128, out_channels=512, stride=2),
            # 512 * 28 * 28
            bottleneck_block(in_channels=512, out_channels1=128, out_channels=512, stride=1),
            # 512 * 28 * 28
            bottleneck_block(in_channels=512, out_channels1=128, out_channels=512, stride=1),
            # 512 * 28 * 28
            bottleneck_block(in_channels=512, out_channels1=128, out_channels=512, stride=1)
            # 512 * 28 * 28
        )
        # 512 * 28 * 28
        self.conv4 = nn.Sequential(
            bottleneck_block(in_channels=512, out_channels1=256, out_channels=1024, stride=2),
            # 1024 * 14 * 14
            bottleneck_block(in_channels=1024, out_channels1=256, out_channels=1024, stride=1),
            # 1024 * 14 * 14
            bottleneck_block(in_channels=1024, out_channels1=256, out_channels=1024, stride=1),
            # 1024 * 14 * 14
            bottleneck_block(in_channels=1024, out_channels1=256, out_channels=1024, stride=1),
            # 1024 * 14 * 14
            bottleneck_block(in_channels=1024, out_channels1=256, out_channels=1024, stride=1),
            # 1024 * 14 * 14
            bottleneck_block(in_channels=1024, out_channels1=256, out_channels=1024, stride=1)
            # 1024 * 14 * 14
        )
        # 1024 * 14 * 14
        self.conv5 = nn.Sequential(
            bottleneck_block(in_channels=1024, out_channels1=512, out_channels=2048, stride=2),
            # 2048 * 7 * 7
            bottleneck_block(in_channels=2048, out_channels1=512, out_channels=2048, stride=1),
            # 2048 * 7 * 7
            bottleneck_block(in_channels=2048, out_channels1=512, out_channels=2048, stride=1)
            # 2048 * 7 * 7
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        # 2048 * 1 * 1
        # fully connection
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048 * 1 * 1, out_features=n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # input: 3 * 224 * 224
        x = self.conv1(x)
        # 64 * 112 * 112
        x = self.conv2(x)
        # 256 * 56 * 56
        x = self.conv3(x)
        # 512 * 28 * 28
        x = self.conv4(x)
        # 1024 * 14 * 14
        x = self.conv5(x)
        # 2048 * 7 * 7
        x = self.avgpool(x)
        # 2048 * 1 * 1
        x = x.view(-1, 2048 * 1 * 1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    x = torch.FloatTensor(1, 3, 224, 224)
    network = resnet_50(n_class=1000)
    x = network(x)
    print(x.size())

