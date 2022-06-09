# -*- coding = utf-8 -*-
# @Time : 2021/12/8 17:49
# @Author : Luxlios
# @File : FPN.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torchvision

class FPN(nn.Module):
    def __init__(self, backbone):
        super(FPN, self).__init__()
        # lateral connection
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)

        # top-down pathway
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # cmooth
        self.smooth4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.init_params()

        # resnet, vgg..
        self.backbone = backbone

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # bottom-up pathway
        x = self.backbone[0](x)
        c2 = x
        x = self.backbone[1](x)
        c3 = x
        x = self.backbone[2](x)
        c4 = x
        c5 = self.backbone[3](x)

        # top-down pathway and lateral connections
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p2 = self.upsample2(p3) + self.lateral2(c2)

        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        return p5, p4, p3, p2

if __name__ == '__main__':
    # backbone
    resnet = torchvision.models.resnet50(pretrained=True)
    layers = list(resnet.children())
    backbone = list()
    backbone.append(nn.Sequential(*layers[:5]))  # / 4
    backbone.append(nn.Sequential(layers[5]))  # / 8
    backbone.append(nn.Sequential(layers[6]))  # / 16
    backbone.append(nn.Sequential(layers[7]))  # / 32
    # freeze
    for block in backbone:
        for param in list(block.parameters()):
            param.requires_grad = False

    network = FPN(backbone=backbone)
    x = torch.FloatTensor(1, 3, 800, 800)
    p5, p4, p3, p2 = network(x)
    print(p5.size())
    print(p4.size())
    print(p3.size())
    print(p2.size())
