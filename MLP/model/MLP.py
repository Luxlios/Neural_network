# -*- coding = utf-8 -*-
# @Time : 2021/12/10 20:09
# @Author : Luxlios
# @File : MLP.py
# @Software : PyCharm

import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, n_class=10):
        super(mlp, self).__init__()
        # input: 3 * 32 * 32 (cifar-10)
        self.fc = nn.Sequential(
            nn.Linear(in_features=32 * 32 * 3, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    x = torch.FloatTensor(1, 3, 32, 32)
    network = mlp(n_class=10)
    x = network(x)
    print(x.size())

