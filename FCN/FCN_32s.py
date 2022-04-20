# -*- coding = utf-8 -*-
# @Time : 2022/4/13 13:47
# @Author : Luxlios
# @File : FCN_32s.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torch.optim as optim
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class fcn32s(nn.Module):
    def __init__(self, vgg16_rmfc, n_class):
        super(fcn32s, self).__init__()
        # input: arbitrary size
        # example: 224 * 224 * 3
        self.vgg16_rmfc = vgg16_rmfc
        # 7 * 7 * 512
        self.fc2conv = nn.Sequential(
            # fc to conv
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # del classifier & append a 1 * 1 convolution to predict each classes(dimension 21)
            nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1, padding=0)
        )
        # 7 * 7 * n_class
        self.deconv2x = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=3,
                               stride=2, padding=1, dilation=1, output_padding=1),
            # Conv2d: o = floor((i + padding - kernel_size) / 2) + 1
            # ConvTranspose2d: input: i'=o, output: o'=i exists 2 values,
            # eliminate by output_padding(=1 larger one, =0 smaller one)

            # padding_trans = kernel_size - 1 - padding
            # input_size = stride * (7 - 1) + 1 + padding_trans * 2
            # output_size = (input_size - kernel_size) / 1 + 1 + output_padding
            # dilation does not have influence(auto padding?)

            # conclusion:
            # ConvTranspose2d.output_size = (input_size - 1) * stride + kernel_size - 2 * padding + output_padding
            nn.BatchNorm2d(n_class),
            nn.ReLU()
        )

        self.deconv2xpred = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=3,
                               stride=2, padding=1, dilation=1, output_padding=1)

        )
        # 224 * 224 * n_class

    def forward(self, x):
        x5 = self.vgg16_rmfc(x)
        # 7 * 7 * 512
        x5 = x5.view(x5.size(0), 512, 7, 7)
        x5 = self.fc2conv(x5)
        # 7 * 7 * n_class

        x = self.deconv2x(x5)
        # 14 * 14 * n_class
        x = self.deconv2x(x)
        # 28 * 28 * n_class
        x = self.deconv2x(x)
        # 56 * 56 * n_class
        x = self.deconv2x(x)
        # 112 * 112 * n_class
        x = self.deconv2xpred(x)
        # 224 * 224 * n_class

        return x


if __name__ == '__main__':
    # create pretrained vgg16 & remove classifier layer
    n_class = 21
    vgg16_rmfc = models.vgg16(pretrained=True)
    vgg16_rmfc.classifier = nn.Sequential()
    # for param in list(vgg16_rmfc.parameters()):
    #     param.requires_grad = False

    network = fcn32s(vgg16_rmfc=vgg16_rmfc, n_class=n_class)
    # optimizer = optim.SGD(filter(lambda x: x.requires_grad, network.parameters()), lr=1e-5)
    x = torch.FloatTensor(1, 3, 224, 224)
    x = network(x)
    print(x.size())


'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential()
)
'''