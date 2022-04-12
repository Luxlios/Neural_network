# -*- coding = utf-8 -*-
# @Time : 2022/4/10 10:29
# @Author : Luxlios
# @File : model_train.py
# @Software : PyCharm

import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from UNet import UNet

def make_txt(path, train_size=1.0, shuffle=True):
    # Divide the dataset into training set and test set and convert it to txt
    # data: path/images/1.png
    # label: path/masks/1.png
    '''
    :param path: Address where data and labels are stored
    :param train_size: Percentage of training data
    :param shuffle: If or not shuffle
    '''
    if os.path.exists(os.path.join(path, 'train.txt')):
        os.remove(os.path.join(path, 'train.txt'))
    if os.path.exists(os.path.join(path, 'test.txt')):
        os.remove(os.path.join(path, 'test.txt'))
    train = open(os.path.join(path, 'train.txt'), 'a')
    test = open(os.path.join(path, 'test.txt'), 'a')

    # images: 1.png
    # masks: 1.png
    images = os.path.join(path, 'images')
    masks = os.path.join(path, 'masks')
    files_images = os.listdir(images)
    if shuffle:
        random.shuffle(files_images)
    else:
        pass
    for i in range(len(files_images)):
        if i < len(files_images) * train_size:
            filename = files_images[i]
            txt_temp = os.path.join(images, filename) + ' ' + os.path.join(masks, filename) + '\n'
            train.write(txt_temp)
        else:
            filename = files_images[i]
            txt_temp = os.path.join(images, filename) + ' ' + os.path.join(masks, filename) + '\n'
            test.write(txt_temp)

    train.close()
    test.close()

def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')  # grayscale

class SegDataset(Dataset):
    def __init__(self, txt, transform_image=None, transform_mask=None, loader=default_loader):
        super(SegDataset, self).__init__()
        txt_content = open(txt, 'r')
        imagenmask = []
        for line in txt_content:
            line = line.rstrip('\n')
            words = line.split(' ')
            imagenmask.append([words[0], words[1]])

        self.imagenmask = imagenmask
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.loader = loader

    def __getitem__(self, index):
        image, mask = self.imagenmask[index]
        img = self.loader(image)
        mas = self.loader(mask)
        if self.transform_image is not None:
            img = self.transform_image(img)
        if self.transform_mask is not None:
            mas = self.transform_mask(mas)
        return img, mas

    def __len__(self):
        return len(self.imagenmask)

if __name__ == '__main__':
    make_txt('tgs-salt-identification-challenge/train', train_size=1, shuffle=True)
    transformer_image = transforms.Compose([transforms.Resize((388, 388)),  # 388 + 92 + 92 =572
                                            transforms.Pad(padding=92, padding_mode='reflect'),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])

    transformer_mask = transforms.Compose([transforms.Resize((388, 388)),
                                           transforms.ToTensor()])

    train_data = SegDataset(txt=os.path.join(os.getcwd(), 'tgs-salt-identification-challenge/train/train.txt'),
                            transform_image=transformer_image, transform_mask=transformer_mask)
    test_data = SegDataset(txt=os.path.join(os.getcwd(), 'tgs-salt-identification-challenge/train/test.txt'),
                           transform_image=transformer_image, transform_mask=transformer_mask)

    print('Num_of_train:', len(train_data))
    print('Num_of_test:', len(test_data))

    batch_size = 15
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # optimization & loss function
    loss_function = nn.BCEWithLogitsLoss()
    # GPU
    device = torch.device("cuda:0")
    network = UNet(n_class=1).to(device)
    network.cuda()
    # network = UNet(n_class=1)
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    # train
    epochs = 100
    accuracy_record_train = []
    accuracy_record_test = []
    for epoch in range(epochs):
        network.train()
        for data in iter(trainloader):
            inputs, labels = data
    #         GPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # classifier
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            # 反向梯度传播
            loss.backward()
            # 用梯度做优化
            optimizer.step()
        print('[epoch: %d] loss: %.3f'%(epoch, loss))
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
            network = inception_v1(n_class=10, state='train')
            # optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.1)
            checkpoint = torch.load('./model.pth.tar')
            network.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            '''
        else:
            pass



