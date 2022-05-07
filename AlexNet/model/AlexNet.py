# -*- coding = utf-8 -*-
# @Time : 2021/12/10 20:20
# @Author : Luxlios
# @File : AlexNet.py
# @Software : PyCharm

import torch
import torch.nn as nn

class alexnet(nn.Module):
    def __init__(self, n_class=10):
        super(alexnet, self).__init__()
        # input: 3 * 32 * 32    paper: 3 * 224 * 224(preprocess->3 * 227 * 227)
        # convolution & pooling
        self.conv = nn.Sequential(
            # paper:kernel_size = 11, stride = 4, padding = 2
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # 96 * 16 * 16
            # paper:kernel_size = 3, stride = 2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 96 * 8 * 8
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # paper:kernel_size = 3, stride = 2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 256 * 4 * 4
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # paper:kernel_size = 3, stride = 2
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 384 * 2 * 2
        )
        # fully connection
        self.fc = nn.Sequential(
            nn.Linear(in_features=2 * 2 * 384, out_features=4096),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=n_class),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 2 * 2 * 384)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.FloatTensor(1, 3, 32, 32)
    network = alexnet(n_class=10)
    x = network(x)
    print(x.size())

