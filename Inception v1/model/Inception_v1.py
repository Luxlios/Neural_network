# -*- coding = utf-8 -*-
# @Time : 2022/04/05 10:09
# @Author : Luxlios
# @File : Inception_v1.py
# @Software : PyCharm

import torch
import torch.nn as nn

class inception(nn.Module):
    def __init__(self, in_chan, out_bran11, out_bran21, out_bran22,
                 out_bran31, out_bran32, out_bran42):
        super(inception, self).__init__()
        # input: n * n * in_chan
        # branch1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran11, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
            # n * n * out_bran11
        )
        # branch2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran21, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # n * n * out_bran21
            nn.Conv2d(in_channels=out_bran21, out_channels=out_bran22, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            # n * n * out_bran22
        )
        # branch3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran31, kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # n+1 * n+1 * out_bran31
            nn.Conv2d(in_channels=out_bran31, out_channels=out_bran32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True)
            # n * n * out_bran32
        )
        #branch4
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            # n-1 * n-1 * in_chan
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran42, kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True)
            # n * n * out_bran42
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        # DepthConcat
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x

class inception_aux(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(inception_aux, self).__init__()
        # inception v1 inputs: in_chan * 14 * 14
        self.conv = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3, padding=0),  # in_chan*4*4
                                  nn.Conv2d(in_channels=in_chan, out_channels=128, kernel_size=1, padding=0),  # 128*4*4
                                  nn.ReLU(inplace=True))

        self.fc = nn.Sequential(nn.Linear(in_features=128 * 4 * 4, out_features=1024),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.7),
                                nn.Linear(in_features=1024, out_features=out_chan),
                                nn.Softmax(dim=1)
                                )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc(x)
        return x

class inception_v1(nn.Module):
    def __init__(self, n_class, state='train'):
        super(inception_v1, self).__init__()
        # input: 32 * 32 * 3    paper: 224 * 224 * 3
        # resize to 224*224*3

        self.state = state
        # convolution & pooling
        self.block123 = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),  # 112*112*64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56*56*64
            # block 2
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),  # 56*56*192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 28*28*192
            # block 3-inception
            inception(in_chan=192, out_bran11=64, out_bran21=96, out_bran22=128,
                      out_bran31=16, out_bran32=32, out_bran42=32),  # 28*28*256
            inception(in_chan=256, out_bran11=128, out_bran21=128, out_bran22=192,
                      out_bran31=32, out_bran32=96, out_bran42=64),  # 28*28*480
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 14*14*480
            )
            # layer 4-inception exist 2 aux-classifier
        self.block4_1 = inception(in_chan=480, out_bran11=192, out_bran21=96, out_bran22=208,
                                  out_bran31=16, out_bran32=48, out_bran42=64)  # 14*14*512 aux_classifier
        if self.state == 'train':
            self.aux1 = inception_aux(in_chan=512, out_chan=n_class)

        self.block4_2 = nn.Sequential(
            inception(in_chan=512, out_bran11=160, out_bran21=112, out_bran22=224,
                      out_bran31=24, out_bran32=64, out_bran42=64),  # 14*14*512
            inception(in_chan=512, out_bran11=128, out_bran21=128, out_bran22=256,
                      out_bran31=24, out_bran32=64, out_bran42=64),  # 14*14*512
            inception(in_chan=512, out_bran11=112, out_bran21=144, out_bran22=288,
                      out_bran31=32, out_bran32=64, out_bran42=64)  # 14*14*528 aux_classifier
            )

        if self.state == 'train':
            self.aux2 = inception_aux(in_chan=528, out_chan=n_class)

        self.block4_3 = nn.Sequential(
            inception(in_chan=528, out_bran11=256, out_bran21=160, out_bran22=320,
                      out_bran31=32, out_bran32=128, out_bran42=128),  # 14*14*832
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 7*7*832
        )

        self.block56 = nn.Sequential(
            # layer 5-inception
            inception(in_chan=832, out_bran11=256, out_bran21=160, out_bran22=320,
                      out_bran31=32, out_bran32=128, out_bran42=128),  # 7*7*832
            inception(in_chan=832, out_bran11=384, out_bran21=192, out_bran22=384,
                      out_bran31=48, out_bran32=128, out_bran42=128),  # 7*7*1024
            # layer 6
            nn.AvgPool2d(kernel_size=7, stride=1)  # 1*1*1024
        )
        # fully connection
        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block123(x)
        x = self.block4_1(x)
        aux_out1 = x
        x = self.block4_2(x)
        aux_out2 = x
        x = self.block4_3(x)
        x = self.block56(x)
        # 1024*1*1 -> 1024
        x = x.view(-1, 1024 * 1 * 1)
        x = self.fc(x)

        if self.state == 'train':
            aux_out1 = self.aux1(aux_out1)
            aux_out2 = self.aux2(aux_out2)
            return aux_out1, aux_out2, x
        else:
            return x

if __name__ == '__main__':
    x = torch.FloatTensor(1, 3, 224, 224)
    network = inception_v1(n_class=10)
    aux_out1, aux_out2, x = network(x)
    print(x.size())
