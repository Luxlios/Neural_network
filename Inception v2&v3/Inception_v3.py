# -*- coding = utf-8 -*-
# @Time : 2022/4/6 16:38
# @Author : Luxlios
# @File : Inception_v3.py
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

class inceptionA(nn.Module):
    def __init__(self, in_chan, out_bran11, out_bran21, out_bran22,
                 out_bran31, out_bran32, out_bran42):
        super(inceptionA, self).__init__()
        # input: n * n * in_chan
        # branch1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran11, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran11),
            nn.ReLU(inplace=True)
            # n * n * out_bran11
        )
        # branch2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran21, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran21),
            nn.ReLU(inplace=True),
            # n * n * out_bran21
            nn.Conv2d(in_channels=out_bran21, out_channels=out_bran22, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_bran22),
            nn.ReLU(inplace=True)
            # n * n * out_bran22
        )
        # branch3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran31, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran31),
            nn.ReLU(inplace=True),
            # n * n * out_bran31
            nn.Conv2d(in_channels=out_bran31, out_channels=out_bran32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_bran32),
            nn.ReLU(inplace=True),
            # n * n * out_bran32
            nn.Conv2d(in_channels=out_bran32, out_channels=out_bran32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_bran32),
            nn.ReLU(inplace=True)
            # n * n * out_bran32
        )
        #branch4
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # n * n * in_chan
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran42, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran42),
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

class inceptionB(nn.Module):
    def __init__(self, in_chan, out_bran11, out_bran21, out_bran22,
                 out_bran31, out_bran32, out_bran42):
        super(inceptionB, self).__init__()
        # input: n * n * in_chan
        # branch1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran11, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran11),
            nn.ReLU(inplace=True)
            # n * n * out_bran11
        )
        # branch2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran21, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran21),
            nn.ReLU(inplace=True),
            # n * n * out_bran21
            nn.Conv2d(in_channels=out_bran21, out_channels=out_bran21, kernel_size=[1, 7], stride=1, padding=[0, 3]),
            nn.BatchNorm2d(num_features=out_bran21),
            nn.ReLU(inplace=True),
            # n * n * out_bran21
            nn.Conv2d(in_channels=out_bran21, out_channels=out_bran22, kernel_size=[7, 1], stride=1, padding=[3, 0]),
            nn.BatchNorm2d(num_features=out_bran22),
            nn.ReLU(inplace=True)
            # n * n * out_bran22
        )
        # branch3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran31, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran31),
            nn.ReLU(inplace=True),
            # n * n * out_bran31
            nn.Conv2d(in_channels=out_bran31, out_channels=out_bran31, kernel_size=[1, 7], stride=1, padding=[0, 3]),
            nn.BatchNorm2d(num_features=out_bran31),
            nn.ReLU(inplace=True),
            # n * n * out_bran31
            nn.Conv2d(in_channels=out_bran31, out_channels=out_bran31, kernel_size=[7, 1], stride=1, padding=[3, 0]),
            nn.BatchNorm2d(num_features=out_bran31),
            nn.ReLU(inplace=True),
            # n * n * out_bran31
            nn.Conv2d(in_channels=out_bran31, out_channels=out_bran31, kernel_size=[1, 7], stride=1, padding=[0, 3]),
            nn.BatchNorm2d(num_features=out_bran31),
            nn.ReLU(inplace=True),
            # n * n * out_bran31
            nn.Conv2d(in_channels=out_bran31, out_channels=out_bran32, kernel_size=[7, 1], stride=1, padding=[3, 0]),
            nn.BatchNorm2d(num_features=out_bran32),
            nn.ReLU(inplace=True),
            # n * n * out_bran32
        )
        #branch4
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # n * n * in_chan
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran42, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran42),
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

class inceptionC(nn.Module):
    def __init__(self, in_chan, out_bran11, out_bran21, out_bran22,
                 out_bran31, out_bran32, out_bran42):
        super(inceptionC, self).__init__()
        # input: n * n * in_chan
        # branch1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran11, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran11),
            nn.ReLU(inplace=True)
            # n * n * out_bran11
        )
        # branch2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran21, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran21),
            nn.ReLU(inplace=True),
            # n * n * out_bran21
        )
        self.branch21 = nn.Sequential(
            nn.Conv2d(in_channels=out_bran21, out_channels=out_bran22, kernel_size=[1, 3], stride=1, padding=[0, 1]),
            nn.BatchNorm2d(num_features=out_bran22),
            nn.ReLU(inplace=True),
            # n * n * out_bran22
        )
        self.branch22 = nn.Sequential(
            nn.Conv2d(in_channels=out_bran21, out_channels=out_bran22, kernel_size=[3, 1], stride=1, padding=[1, 0]),
            nn.BatchNorm2d(num_features=out_bran22),
            nn.ReLU(inplace=True)
            # n * n * out_bran22
        )
        # branch3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran31, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran31),
            nn.ReLU(inplace=True),
            # n * n * out_bran21
            nn.Conv2d(in_channels=out_bran31, out_channels=out_bran32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_bran32),
            nn.ReLU(inplace=True),
            # n * n * out_bran21
        )
        self.branch31 = nn.Sequential(
            nn.Conv2d(in_channels=out_bran32, out_channels=out_bran32, kernel_size=[1, 3], stride=1, padding=[0, 1]),
            nn.BatchNorm2d(num_features=out_bran32),
            nn.ReLU(inplace=True),
            # n * n * out_bran32
        )
        self.branch32 = nn.Sequential(
            nn.Conv2d(in_channels=out_bran32, out_channels=out_bran32, kernel_size=[3, 1], stride=1, padding=[1, 0]),
            nn.BatchNorm2d(num_features=out_bran32),
            nn.ReLU(inplace=True)
            # n * n * out_bran32
        )
        #branch4
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # n * n * in_chan
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran42, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran42),
            nn.ReLU(inplace=True)
            # n * n * out_bran42
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x2 = torch.cat([self.branch21(x2), self.branch22(x2)], dim=1)
        x3 = self.branch3(x)
        x3 = torch.cat([self.branch31(x3), self.branch32(x3)], dim=1)
        x4 = self.branch4(x)

        # DepthConcat
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x

class inceptionD(nn.Module):
    def __init__(self, in_chan, out_bran11, out_bran12, out_bran21, out_bran22):
        super(inceptionD, self).__init__()
        # input: n * n * in_chan
        # branch1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran11, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran11),
            nn.ReLU(inplace=True),
            # n * n * out_bran11
            nn.Conv2d(in_channels=out_bran11, out_channels=out_bran12, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=out_bran12),
            nn.ReLU(inplace=True)
            # n/2 * n/2 * out_bran12

        )
        # branch2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_bran21, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_bran21),
            nn.ReLU(inplace=True),
            # n * n * out_bran21
            nn.Conv2d(in_channels=out_bran21, out_channels=out_bran22, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_bran22),
            nn.ReLU(inplace=True),
            # n * n * out_bran22
            nn.Conv2d(in_channels=out_bran22, out_channels=out_bran22, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=out_bran22),
            nn.ReLU(inplace=True)
            # n/2 * n/2 * out_bran22
        )
        #branch3
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # n/2 * n/2 * in_chan

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # DepthConcat
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class inception_aux(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(inception_aux, self).__init__()
        # inception v2 inputs: in_chan * 17 * 17(located in last 17 * 17 layer)
        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
            # in_chan*5*5
            nn.Conv2d(in_channels=in_chan, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # 128*5*5
        )

        self.fc = nn.Sequential(nn.Linear(in_features=128 * 5 * 5, out_features=1024),
                                nn.BatchNorm1d(num_features=1024),    # Auxiliary classifier + BN
                                nn.ReLU(),
                                nn.Linear(in_features=1024, out_features=out_chan),
                                nn.Softmax(dim=1)
                                )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * 5 * 5)
        x = self.fc(x)
        return x

# Label-Smoothing Regularization as Loss function(Calculate Cross-Entropy loss after label smoothing)
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # smooth_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / num_classes

    def forward(self, x, target):
        logprob = F.log_softmax(x, dim=-1)
        nll_loss = -logprob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class inception_v3(nn.Module):
    def __init__(self, n_class, state='train'):
        super(inception_v3, self).__init__()
        # input: 32 * 32 * 3    paper: 299 * 299 * 3
        # resize to 299*299*3

        self.state = state
        # convolution & pooling
        self.block12 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            # 149 * 149 *32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            # 147 * 147 *32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # 147 * 147 * 64
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 73 * 73 * 64
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=80),
            nn.ReLU(inplace=True),
            # 71 * 71 * 80
            nn.Conv2d(in_channels=80, out_channels=192, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            # 35 * 35 * 192
            nn.Conv2d(in_channels=192, out_channels=288, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=288),
            nn.ReLU(inplace=True),
            # 35 * 35 * 288
        )
        self.block3 = nn.Sequential(
            inceptionA(in_chan=288, out_bran11=64, out_bran21=48, out_bran22=64,
                       out_bran31=64, out_bran32=96, out_bran42=32),
            # 35 * 35 * 256
            inceptionA(in_chan=256, out_bran11=64, out_bran21=48, out_bran22=64,
                       out_bran31=64, out_bran32=96, out_bran42=64),
            # 35 * 35 * 288
            inceptionD(in_chan=288, out_bran11=384, out_bran12=384, out_bran21=64, out_bran22=96)
            # 17 * 17 * 768
        )
        self.block41 = nn.Sequential(
            inceptionB(in_chan=768, out_bran11=192, out_bran21=128, out_bran22=192,
                       out_bran31=128, out_bran32=192, out_bran42=192),
            # 17 * 17 * 768
            inceptionB(in_chan=768, out_bran11=192, out_bran21=160, out_bran22=192,
                       out_bran31=160, out_bran32=192, out_bran42=192),
            # 17 * 17 * 768
            inceptionB(in_chan=768, out_bran11=192, out_bran21=160, out_bran22=192,
                       out_bran31=160, out_bran32=192, out_bran42=192),
            # 17 * 17 * 768
            inceptionB(in_chan=768, out_bran11=192, out_bran21=192, out_bran22=192,
                       out_bran31=192, out_bran32=192, out_bran42=192),
            # 17 * 17 * 768
        )

        if self.state == 'train':
            self.aux = inception_aux(in_chan=768, out_chan=n_class)

        self.block42 = nn.Sequential(
            inceptionD(in_chan=768, out_bran11=192, out_bran12=320, out_bran21=192, out_bran22=192)
            # 8 * 8 * 1280
        )
        self.block5 = nn.Sequential(
            inceptionC(in_chan=1280, out_bran11=320, out_bran21=384, out_bran22=384,
                       out_bran31=448, out_bran32=384, out_bran42=192),
            # inceptionC: branch 2 & 3 have 2 side branches
            # 8 * 8 * 2048
            inceptionC(in_chan=2048, out_bran11=320, out_bran21=384, out_bran22=384,
                       out_bran31=448, out_bran32=384, out_bran42=192),
            # 8 * 8 * 2048
            nn.MaxPool2d(kernel_size=8, stride=1, padding=0),
            # 1 * 1 * 2048
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=n_class),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.block12(x)
        x = self.block3(x)
        x = self.block41(x)
        aux_out = x
        x = self.block42(x)
        x = self.block5(x)
        # 2048*1*1 -> 2048
        x = x.view(-1, 2048 * 1 * 1)
        x = self.fc(x)

        if self.state == 'train':
            aux_out = self.aux(aux_out)
            return aux_out, x
        else:
            return x

if __name__ == '__main__':
    # 读取数据集cifar

    # generalization
    # 把rgb值标准化到均值为0.5，方差为0.5，归一化到[0, 1]这个区间
    transformer = transforms.Compose([transforms.Resize((299, 299)),
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
    trainloader = DataLoader(cifar_train, batch_size=32, shuffle=True)
    testloader = DataLoader(cifar_test, batch_size=32, shuffle=True)

    # optimization & loss function
    loss_function = nn.CrossEntropyLoss()
    # GPU
    # device = torch.device("cuda:0")
    # network = inception_v3(n_class=10, state='train').to(device)
    # network.cuda()
    network = inception_v3(n_class=10)

    optimizer = optim.RMSprop(network.parameters(), lr=0.045, weight_decay=0.9, eps=1.0)
    # lr exponential decayed
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)

    # train & test
    lsr = LabelSmoothing(smoothing=0.1)  # Label-Smoothing Regularization(cross-entropy loss function after label smoothing)
    epoches = 25
    accuracy_record_train = []
    accuracy_record_test = []
    for epoch in range(epoches):
        network.train()
        for data in iter(trainloader):
            inputs, labels = data
    #         GPU
    #         inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # auxiliary classifier + LabelSmoothingRegularization
            aux_out, outputs = network(inputs)
            # loss = loss_function(outputs, labels)
            # loss_aux = loss_function(aux_out, labels)
            loss = lsr(outputs, labels)
            loss_aux = lsr(aux_out, labels)
            loss = loss + 0.3 * loss_aux  # paper do not mention weight
            # 反向梯度传播
            loss.backward()
            # gradient clipping（梯度裁剪，裁剪到某个范围内，防止梯度爆炸）
            threshold = 2
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=threshold, norm_type=2)
            # 用梯度做优化
            optimizer.step()

        # lr decayed every two epoch
        if epoch == 0:
            pass
        elif (epoch + 1) // 2 == 0:
            scheduler.step()
        else:
            pass

        # train精度
        total = 0
        correction = 0
        network.eval()
        with torch.no_grad():
            for data in iter(trainloader):
                inputs, labels = data
    #             GPU
    #             inputs, labels = inputs.to(device), labels.to(device)
                aux_out, outputs = network(inputs)
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
    #             inputs, labels = inputs.to(device), labels.to(device)
                aux_out, outputs = network(inputs)
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
    network = inception_v2(n_class=10, state='train')
    # optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.1)
    checkpoint = torch.load('./model.pth.tar')
    network.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    '''
