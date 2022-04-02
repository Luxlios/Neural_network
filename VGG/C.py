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


class vggC(nn.Module):
    def __init__(self, n_class):
        super(vggC, self).__init__()
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
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4*4*256
            # layer 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2*2*512
            # layer 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1),
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
    # 读取数据集cifar

    # generalization
    # 把rgb值标准化到均值为0.5，方差为0.5，归一化到[0, 1]这个区间
    transformer = transforms.Compose([transforms.ToTensor(),
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
    # device = torch.device("cuda:0")
    # network = vggC(n_class=10).to(device)
    network = vggC(n_class=10)
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
    #         inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
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
    #             inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
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
                outputs = network(inputs)
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

