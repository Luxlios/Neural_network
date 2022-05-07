# -*- coding = utf-8 -*-
# @Time : 2021/12/10 20:09
# @Author : Luxlios
# @File : main.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model.MLP import mlp
import logging
import os
import sys
import datetime
import logging
import shutil
from torch.utils.tensorboard import SummaryWriter

def logger_init(log_file_name='monitor', log_level=logging.DEBUG, log_dir='./logs/', only_file=False):
    # log path
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.datetime.now())[:10] + '_' +
                            str(datetime.datetime.now())[11:19].replace(':', '-') + '.log')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        # only output info to local file
        logging.basicConfig(filename=log_path, level=log_level, format=formatter, datefmt='%Y-%d-%m %H:%M:%S')
    else:
        # simultaneously output info to local file and terminal
        logging.basicConfig(level=log_level, format=formatter, datefmt='%Y-%d-%m %H:%M:%S',
                            handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])

if __name__ == '__main__':
    # load cifar from torchvision.datasets
    # generalization
    # r, g, b three dimensions mean to 0.5, 0.5, 0.5  standard deviation to 0.5, 0.5, 0.5
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])
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
    # image_data: np.array
    # targets: 一个list

    # dataloader
    trainloader = DataLoader(cifar_train, batch_size=50, shuffle=True)
    testloader = DataLoader(cifar_test, batch_size=50, shuffle=True)

    # optimization & loss function
    loss_function = nn.CrossEntropyLoss()
    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = mlp(n_class=10).to(device)
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    # train & test
    # tensorboard log
    # if os.path.exists('./tb-log'):
    #     shutil.rmtree('./tb-log')
    tb = SummaryWriter('./tb-log')
    logger_init(log_file_name='monitor', log_level=logging.DEBUG, log_dir='./logs')
    epoches = 25
    for epoch in range(epoches):
        network.train()
        loss_total = 0
        for data in iter(trainloader):
            inputs, labels = data
            # GPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            loss_total += loss
            # loss back propagation
            loss.backward()
            # optimized by sgd
            optimizer.step()
        # train accuracy
        total = 0
        correction = 0
        network.eval()
        with torch.no_grad():
            for data in iter(trainloader):
                inputs, labels = data
                # GPU
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                probability, prediction = torch.max(outputs, 1)
                total += labels.size(0)
                correction += (prediction == labels).sum().item()
            accuracy_train = correction / total
            loss = loss_total / total
        # test accuracy
        total = 0
        correction = 0
        network.eval()
        with torch.no_grad():
            for data in iter(testloader):
                inputs, labels = data
                # GPU
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                probability, prediction = torch.max(outputs, 1)
                total += labels.size(0)
                correction += (prediction == labels).sum().item()
            accuracy_test = correction / total
        # log
        tb.add_scalar(tag='loss',
                      scalar_value=loss,
                      global_step=epoch)
        tb.add_scalars(main_tag='accuracy',
                       tag_scalar_dict={'accuracy_train': accuracy_train,
                                        'accuracy_test': accuracy_test},
                       global_step=epoch)
        logging.info(f'[epoch {epoch + 1: d}]loss:{loss: .5f}, train_accuracy:{accuracy_train * 100: .5f}%, '
                     f'test_accuracy:{accuracy_test * 100: .5f}%')


