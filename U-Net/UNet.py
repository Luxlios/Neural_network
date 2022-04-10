# -*- coding = utf-8 -*-
# @Time : 2022/4/9 11:19
# @Author : Luxlios
# @File : UNet.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def crop(tensor, crop_H, crop_W):
    # crop tensor  (copy and crop)
    # [B, C, H, W]
    H = tensor.size(2)
    W = tensor.size(3)

    top = int((H - crop_H) / 2)
    left = int((W - crop_W) / 2)

    tensor_crop = torchvision.transforms.functional.crop(tensor, top=top, left=left, height=crop_H, width=crop_W)
    return tensor_crop

class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
        # input: 572 * 572 * 1
        # output: 388 * 388 * 2 (分背景和目标，双channels，[1, 0]背景，[0, 1]目标)
        # left
        self.conv_l1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 570 * 570 * 64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 568 * 568 * 64
        )
        self.conv_l2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 284 * 284 * 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 282 * 282 * 128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 280 * 280 * 128
        )
        self.conv_l3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 140 * 140 * 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 138 * 138 * 256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 136 * 136 * 256
        )
        self.conv_l4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 68 * 68 * 256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 66 * 66 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 64 * 64 * 512
        )

        self.conv_bottom = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 32 * 32 * 512
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 30 * 30 * 1024
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 28 * 28 * 1024
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
            # 56 * 56 * 512
        )

        # right
        self.conv_r1 = nn.Sequential(
            # crop & cat
            # input: 56 * 56 * 1024
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 54 * 54 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 52 * 52 * 512
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False)
            # 104 * 104 * 256
        )
        self.conv_r2 = nn.Sequential(
            # crop & cat
            # input: 104 * 104 * 512
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 102 * 102 * 256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 100 * 100 * 256
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=False)
            # 200 * 200 * 128
        )
        self.conv_r3 = nn.Sequential(
            # crop & cat
            # input: 200 * 200 * 256
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 198 * 198 * 128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 196 * 196 * 128
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0, bias=False)
            # 392 * 392 * 64
        )
        self.conv_r4 = nn.Sequential(
            # crop & cat
            # input: 392 * 392 * 128
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 390 * 390 * 64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # 388 * 388 * 64
            # one channel <-> one class <-> one color
            nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=1, stride=1, padding=0)
            # 388 * 388 * n_class
        )
    def forward(self, x):
        # left
        x = self.conv_l1(x)
        x1 = x
        # x1 = x1.resize(392, 392)  # crop
        x1 = crop(tensor=x1, crop_H=392, crop_W=392)
        x = self.conv_l2(x)
        x2 = x
        # x2 = x2.resize(200, 200)  # crop
        x2 = crop(tensor=x2, crop_H=200, crop_W=200)
        x = self.conv_l3(x)
        x3 = x
        # x3 = x3.resize(104, 104)  # crop
        x3 = crop(tensor=x3, crop_H=104, crop_W=104)
        x = self.conv_l4(x)
        x4 = x
        # x4 = x4.resize(56, 56)  # crop
        x4 = crop(tensor=x4, crop_H=56, crop_W=56)

        x = self.conv_bottom(x)

        # right
        x = torch.cat([x, x4], dim=1)
        x = self.conv_r1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_r2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_r3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_r4(x)
        return x


if __name__ == '__main__':
    x = torch.FloatTensor(1, 1, 572, 572)
    network = UNet(n_class=2)
    x = network(x)
    print(x.size())
