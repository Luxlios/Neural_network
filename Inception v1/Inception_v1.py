# -*- coding = utf-8 -*-
# @Time : 2022/04/05 10:09
# @Author : Luxlios
# @File : Inception_v1.py
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
    def __init__(self, n_class, state):
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
    # 读取数据集cifar

    # generalization
    # 把rgb值标准化到均值为0.5，方差为0.5，归一化到[0, 1]这个区间
    transformer = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])

    # 读取数据
    # train-set:50000
    # test-set:10000
    # class:10
    # pixel:32*32*3
    cifar_train = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                               download=True, transform=transformer)
    cifar_test = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                              download=True, transform=transformer)

    # print(cifar_train.data.shape)
    # print(np.array(cifar_train.targets).shape)
    # data是一个np.array
    # targets是一个list

    # dataloader
    trainloader = DataLoader(cifar_train, batch_size=50, shuffle=True)
    testloader = DataLoader(cifar_test, batch_size=50, shuffle=True)

    # optimization & loss function
    loss_function = nn.CrossEntropyLoss()
    # GPU
#     device = torch.device("cuda:0")
#     network = inception_v1(n_class=10, state='train').to(device)
#     network.cuda()
    network = inception_v1(n_class=10)
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    # train & test
    epoches = 25
    accuracy_record_train = []
    accuracy_record_test = []
    for epoch in range(epoches):
        network.train()
        for data in iter(trainloader):
            inputs, labels = data
    #         GPU
#             inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # auxiliary classifier
            aux_out1, aux_out2, outputs = network(inputs)
            loss = loss_function(outputs, labels)
            loss1 = loss_function(aux_out1, labels)
            loss2 = loss_function(aux_out2, labels)
            loss = loss + 0.3*loss1 + 0.3*loss2  # paper中提到were weighted by 0.3
            # 反向梯度传播
            loss.backward()
            # 用梯度做优化
            optimizer.step()
        # train精度
        total = 0
        correction = 0
        network.eval()
        with torch.no_grad():
            for data in iter(trainloader):
                inputs, labels = data
    #             GPU
#                 inputs, labels = inputs.to(device), labels.to(device)
                aux_out1, aux_out2, outputs = network(inputs)
                probability, prediction = torch.max(outputs, 1)
                total += labels.size(0)
                correction += (prediction == labels).sum().item()
            accuracy_train = correction / total
            accuracy_record_train.append(accuracy_train)
        # test精度
        total = 0
        correction = 0
        network.eval()
        with torch.no_grad():
            for data in iter(testloader):
                inputs, labels = data
    #             GPU
#                 inputs, labels = inputs.to(device), labels.to(device)
                aux_out1, aux_out2, outputs = network(inputs)
                probability, prediction = torch.max(outputs, 1)
                total += labels.size(0)
                correction += (prediction == labels).sum().item()
            accuracy_test = correction / total
            accuracy_record_test.append(accuracy_test)
        print('[epoch %d]train_accuracy:%.3f%%  test_accuracy:%.3f%%'%(epoch + 1, accuracy_train * 100, accuracy_test * 100))

    # figure
    plt.figure()
    plt.plot(range(1, epoches + 1), accuracy_record_train, c='C1')
    plt.plot(range(1, epoches + 1), accuracy_record_test, c='C0')
    plt.legend(['train_accuracy', 'test_accuracy'])
    plt.title('accuracy(epoch=25)')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    '''
    # save model
    save_dict = {
        'model': network.state_dict(),
        # 'optimizer': optimizer.state_dict()    # Adam等优化器需要用
    }
    torch.save(save_dict, './model.pth.tar')

    # read model
    network = inception_v1(n_class=10, state='train')
    # optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.1)
    checkpoint = torch.load('./model.pth.tar')
    network.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    '''
