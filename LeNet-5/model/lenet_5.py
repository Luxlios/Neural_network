# -*- coding = utf-8 -*-
# @Time : 2021/12/10 20:19
# @Author : Luxlios
# @File : LeNet-5.py
# @Software : PyCharm

import torch
import torch.nn as nn

class lenet_5(nn.Module):
    def __init__(self, n_class=10):
        super(lenet_5, self).__init__()
        # input:32*32*3(CIFAR-10)
        # convolution & pooling
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fully connection
        self.fc = nn.Sequential(
            nn.Linear(in_features=5 * 5 * 16, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 5 * 5 * 16)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    x = torch.FloatTensor(1, 3, 32, 32)
    network = lenet_5(n_class=10)
    x = network(x)
    print(x.size())