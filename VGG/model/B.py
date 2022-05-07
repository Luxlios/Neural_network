# -*- coding = utf-8 -*-
# @Time : 2022/01/10 20:19
# @Author : Luxlios
# @File : B.py
# @Software : PyCharm
import torch
import torch.nn as nn

class vggB(nn.Module):
    def __init__(self, n_class):
        super(vggB, self).__init__()
        # input:32*32*3 paper:224*224*3(preprocess->227*227*3)
        # convolution & pooling
        self.conv =nn.Sequential(
            # layer 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16*16*64
            # layer 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8*8*128
            # layer 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4*4*256
            # layer 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2*2*512
            # layer 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1*1*512
        )
        #fully connection
        self.fc = nn.Sequential(
            nn.Linear(in_features=1*1*512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=n_class)

        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1*1*512)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    x = torch.FloatTensor(1, 3, 32, 32)
    network = vggA(n_class=10)
    x = network(x)
    print(x.size())

