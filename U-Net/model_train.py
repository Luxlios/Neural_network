# -*- coding = utf-8 -*-
# @Time : 2022/4/10 10:29
# @Author : Luxlios
# @File : model_train.py
# @Software : PyCharm

import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from segmentation_metrics import metrics
from UNet import UNet
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def make_txt(path, train_size=0.7, shuffle=True):
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

def colormap2label():
    COLORMAP = [[0, 0, 0], [255, 255, 255]]
    # 2 classes
    # CLASS = ['sediment', 'salt']

    # torch.long <-> index(index get)  torch.uint8 <-> mask(0 mask, 1 get)
    # tensor index -- torch.long, not the same as numpy
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    # 0xffffff(256 256 256)
    # R,G,B -> [0, 255]
    for i, colormap in enumerate(COLORMAP):
        # Hex -> Dec
        # index: color
        # value: label
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

# 3 channels RGB image -> n_class channels onehot matrix
def mask2labelnonehot(mask, n_class=2, colormap2label=colormap2label()):
    # mask: tensor, channel=3 * height * width

    # mask to label
    # every pixel(3 channels) <-> colormap
    # transforms.ToTensor() --> RGB value / 255
    index_matrix = ((mask[0, :, :] * 255) * 256 + (mask[1, :, :] * 255)) * 256 + (mask[2, :, :] * 255)
    label_matrix = colormap2label[index_matrix.long()]

    # label to onehot
    onehot = F.one_hot(label_matrix, num_classes=n_class)
    onehot = onehot.permute([2, 0, 1]).float()  # torch.long to torch.float

    return label_matrix, onehot

def onehot2labelnmask(onehot):
    # onehot: tensor, batch * channel=21 * height * width

    # onehot to label
    label = torch.max(onehot, 1)[1]

    # label to mask
    COLORMAP = [[0, 0, 0], [255, 255, 255]]
    COLORMAP = torch.Tensor(COLORMAP)
    mask = COLORMAP[label].permute([0, 3, 1, 2])
    return label, mask

def default_imgloader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')   # grayscale
def default_masloader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as  img:
            return img.convert('RGB')

class SegDataset(Dataset):
    def __init__(self, txt, transform_image=None, transform_mask=None, imgloader=default_imgloader,
                 masloader=default_masloader, mask2onehot=mask2labelnonehot):
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
        self.imgloader = imgloader
        self.masloader = masloader
        self.mask2onehot=mask2onehot

    def __getitem__(self, index):
        image, mask = self.imagenmask[index]
        img = self.imgloader(image)
        mas = self.masloader(mask)
        if self.transform_image is not None:
            img = self.transform_image(img)
        if self.transform_mask is not None:
            mas = self.transform_mask(mas)
        label, onehot = self.mask2onehot(mas)
        return img, onehot

    def __len__(self):
        return len(self.imagenmask)

if __name__ == '__main__':
    make_txt('tgs-salt-identification-challenge/train', train_size=0.7, shuffle=True)
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

    batch_size = 20
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # optimization & loss function
    loss_function = nn.BCEWithLogitsLoss()
    # GPU
    device = torch.device("cuda:0")
    n_class=2
    network = UNet(n_class=n_class).to(device)
    network.cuda()
    # network = UNet(n_class=n_class)
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    # train
    epochs = 100
    for epoch in range(epochs):
        network.train()
        network.train()
        for data in iter(trainloader):
            images, onehots = data
            # GPU
            images, onehots = images.to(device), onehots.to(device)
            optimizer.zero_grad()
            # classifier
            outputs = network(images)
            loss = loss_function(outputs, onehots)
            # 反向梯度传播
            loss.backward()
            # 用梯度做优化
            optimizer.step()
        network.eval()
        confusionmatrix = np.zeros([n_class, n_class])
        total = 0
        loss_total = 0
        with torch.no_grad():
            for data in iter(testloader):
                images, onehots = data
                # GPU
                images, onehots = images.to(device), onehots.to(device)
                outputs = network(images)
                loss_total += loss_function(outputs, onehots)
                total += outputs.size(0)
                label_pred, mask_pred = onehot2labelnmask(outputs)
                label_true, mask_true = onehot2labelnmask(onehots)
                label_pred, label_true = label_pred.cpu().numpy().reshape(-1), label_true.cpu().numpy().reshape(-1)
                # calculate confusion matrix
                confusionmatrix += confusion_matrix(label_true, label_pred)
        metric = metrics(n_class=n_class, confusion_matrix=confusionmatrix)
        print('[test epoch: %d]loss: %.3f, pixel acc.: %.3f%%, mean acc.: %.3f%%, mean IU: %.3f%%, f.w. IU: %.3f%%'
              % (epoch + 1, loss_total / total, metric.pixelaccuracy() * 100, metric.meanaccuracy() * 100,
                 metric.meaniu() * 100, metric.frequencyweightediu() * 100))
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
