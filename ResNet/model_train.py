# -*- coding = utf-8 -*-
# @Time : 2022/4/22 18:06
# @Author : Luxlios
# @File : model_train.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from ResNet_50 import resnet_50

def make_txt(path, train_size=0.7, shuffle=True):
    # Divide the dataset into training set and test set and convert it to txt
    # data: path/classname/imagename.png
    # label: path/classname/imagename.png
    '''
    :param path: Address where data and labels are stored
    :param train_size: Percentage of training data
    :param shuffle: If or not shuffle
    '''

    if os.path.exists(os.path.join(path, 'train.txt')):
        os.remove(os.path.join(path, 'train.txt'))
    if os.path.exists(os.path.join(path, 'test.txt')):
        os.remove(os.path.join(path, 'test.txt'))
    class_names = os.listdir(path)
    train = open(os.path.join(path, 'train.txt'), 'a')
    test = open(os.path.join(path, 'test.txt'), 'a')

    for i in range(len(class_names)):
        class_name = class_names[i]
        images = os.path.join(path, class_name)
        files_images = os.listdir(images)
        if shuffle:
            random.shuffle(files_images)
        else:
            pass

        for j in range(len(files_images)):
            if j < len(files_images) * train_size:
                filename = files_images[i]
                txt_temp = os.path.join(images, filename) + ' ' + str(i) + '\n'
                train.write(txt_temp)
            else:
                filename = files_images[i]
                txt_temp = os.path.join(images, filename) + ' ' + str(i) + '\n'
                test.write(txt_temp)
    train.close()
    test.close()


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


if __name__ == '__main__':
    # make dataset to txt file and divide into train and test set
    dataset_path = './imagenet100'
    make_txt(dataset_path)

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
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32, shuffle=False)

    # optimization & loss function
    loss_function = nn.CrossEntropyLoss()
    # GPU
    # device = torch.device("cuda:0")
    # network = resnet_50(n_class=100).to(device)
    network = resnet_50(n_class=100)
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    # train & test
    epoches = 100
    accuracy_record_train = []
    accuracy_record_test = []
    for epoch in range(epoches):
        network.train()
        for data in iter(trainloader):
            inputs, labels = data
            # GPU
            # inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            # loss back propagation
            loss.backward()
            # optimize by gradient
            optimizer.step()
        # train accuracy
        total = 0
        correction = 0
        network.eval()
        with torch.no_grad():
            for data in iter(trainloader):
                inputs, labels = data
                # GPU
                # inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                probability, prediction = torch.max(outputs, 1)
                total += labels.size(0)
                correction += (prediction == labels).sum().item()
            accuracy_train = correction / total
            accuracy_record_train.append(accuracy_train)
        # test accuracy
        total = 0
        correction = 0
        network.eval()
        with torch.no_grad():
            for data in iter(testloader):
                inputs, labels = data
                # GPU
                # inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                probability, prediction = torch.max(outputs, 1)
                total += labels.size(0)
                correction += (prediction == labels).sum().item()
            accuracy_test = correction / total
            accuracy_record_test.append(accuracy_test)
        print('[epoch %d]train_accuracy:%.3f%%  test_accuracy:%.3f%%' % (
        epoch + 1, accuracy_train * 100, accuracy_test * 100))
        # save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            # save model
            save_dict = {
                'model': network.state_dict(),
                # 'optimizer': optimizer.state_dict()    # Adam等优化器需要用
            }
            torch.save(save_dict, './model_' + 'epoch' + str(epoch + 1) + '.pth.tar')
            '''
            # read model
            network = resnet_50(n_class=10, state='train')
            # optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.1)
            checkpoint = torch.load('./model.pth.tar')
            network.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            '''
        else:
            pass
    # figure
    plt.figure()
    plt.plot(range(1, epoches + 1), accuracy_record_train, c='C1')
    plt.plot(range(1, epoches + 1), accuracy_record_test, c='C0')
    plt.legend(['train_accuracy', 'test_accuracy'])
    plt.title('accuracy(epoch=25)')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


