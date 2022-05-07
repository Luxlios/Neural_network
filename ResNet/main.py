# -*- coding = utf-8 -*-
# @Time : 2022/04/22 20:19
# @Author : Luxlios
# @File : main.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from PIL import Image
from model.ResNet_50 import resnet_50
import random
import logging
import os
import sys
import datetime
import logging
import json
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def make_txt(path, train_size=0.7, shuffle=True):
    # Divide the dataset into training set and test set and convert it to txt
    # data: path/classname/imagename.png
    # label: path/classname/imagename.png
    '''
    :param path: Address where data and labels are stored
    :param train_size: Percentage of training data
    :param shuffle: If or not shuffle
    :param return: class and its label dictionary
    '''

    if os.path.exists(os.path.join(path, 'train.txt')):
        os.remove(os.path.join(path, 'train.txt'))
    if os.path.exists(os.path.join(path, 'test.txt')):
        os.remove(os.path.join(path, 'test.txt'))
    class_names = os.listdir(path)
    train = open(os.path.join(path, 'train.txt'), 'a')
    test = open(os.path.join(path, 'test.txt'), 'a')

    class_dic = {}
    for i in range(len(class_names)):
        class_name = class_names[i]
        class_dic[i] = class_name
        images = os.path.join(path, class_name)
        files_images = os.listdir(images)
        if shuffle:
            random.shuffle(files_images)
        else:
            pass

        for j in range(len(files_images)):
            if j < len(files_images) * train_size:
                filename = files_images[j]
                txt_temp = os.path.join(images, filename) + ' ' + str(i) + '\n'
                train.write(txt_temp)
            else:
                filename = files_images[j]
                txt_temp = os.path.join(images, filename) + ' ' + str(i) + '\n'
                test.write(txt_temp)
    train.close()
    test.close()

    return class_dic


# image default loader
def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# create own MyDataset, inherit torch.utils.data.Dataset
class MyDataset(Dataset):
    # initiation
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        # load txt
        txt_content = open(txt, 'r')
        pathnlabel = []
        for line in txt_content:
            # delete right \n
            line = line.rstrip('\n')
            # divide into path & label
            words = line.split(' ')
            pathnlabel.append((words[0], int(words[1])))
        # words[0] -- path，words[1] -- lable
        self.pathnlabel = pathnlabel
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    # get item
    def __getitem__(self, index):
        path, label = self.pathnlabel[index]
        img = self.loader(path)
        if self.transform is not None:
            # transform data
            img = self.transform(img)
        return img, label

    # get len
    def __len__(self):
        return len(self.pathnlabel)


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
    # make dataset to txt file and divide into train and test set
    dataset_path = './imagenet100'
    class_dic = make_txt(dataset_path)

    # save class_dic to json file
    dic_json = json.dumps(class_dic, sort_keys=False, indent=4, separators=(',', ':'))
    f_json = open('class_dic.json', 'w')
    f_json.write(dic_json)
    f_json.close()
    '''
    # load json to dic
    f_json = open('class_dic.json', 'r')
    class_dic = json.load(f_json)
    f_json.close()
    '''

    # iterative machine
    transformer = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])
    train_data = MyDataset(txt=os.path.join(dataset_path, 'train.txt'), transform=transformer)
    test_data = MyDataset(txt=os.path.join(dataset_path, 'test.txt'), transform=transformer)

    print('Num_of_train:', len(train_data))
    print('Num_of_test:', len(test_data))

    # dataloader
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # optimization & loss function
    loss_function = nn.CrossEntropyLoss()
    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = resnet_50(n_class=100).to(device)
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    # train & test
    # tensorboard log
    # if os.path.exists('./tb-log'):
    #     shutil.rmtree('./tb-log')
    tb = SummaryWriter('./tb-log')
    logger_init(log_file_name='monitor', log_level=logging.INFO, log_dir='./logs')
    # checkpoint
    if not os.path.exists('./checkpoint'):
        os.mkdir('checkpoint')
    epoches = 25
    for epoch in range(epoches):
        network.train()
        loss_total = 0
        for data in tqdm(iterable=trainloader, desc='epoch {}'.format(epoch+1), ncols=80):
            inputs, labels = data
            # GPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
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
        # save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            # save model
            save_dict = {
                'model': network.state_dict(),
                # 'optimizer': optimizer.state_dict()    # Adam等优化器需要用
            }
            torch.save(save_dict, './checkpoint/model_' + 'epoch' + str(epoch + 1) + '.pth.tar')
            '''
            # read model
            network = inception_v1(n_class=10, state='train')
            # optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.1)
            checkpoint = torch.load('./checkpoint/model.pth.tar')
            network.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            '''
        else:
            pass


