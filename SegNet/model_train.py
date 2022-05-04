# -*- coding = utf-8 -*-
# @Time : 2022/5/3 19:30
# @Author : Luxlios
# @File : model_train.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from SegNet import encoder, segnet
import os
from PIL import Image
import random
import numpy as np
from sklearn.metrics import confusion_matrix
from segmentation_metrics import metrics

def make_txt(path):
    # generate txt about train, text and val dataset
    '''
    :param path: Address where data and labels are stored
    '''
    if os.path.exists(os.path.join(path, 'train.txt')):
        os.remove(os.path.join(path, 'train.txt'))
    if os.path.exists(os.path.join(path, 'val.txt')):
        os.remove(os.path.join(path, 'val.txt'))
    if os.path.exists(os.path.join(path, 'test.txt')):
        os.remove(os.path.join(path, 'test.txt'))
    train = open(os.path.join(path, 'train.txt'), 'a')
    val = open(os.path.join(path, 'val.txt'), 'a')
    test = open(os.path.join(path, 'test.txt'), 'a')

    # images: 1.png
    # masks: 1_L.png
    images_train = os.path.join(path, 'camvid_train')
    images_val = os.path.join(path, 'camvid_val')
    images_test = os.path.join(path,'camvid_test')
    masks_train = os.path.join(path, 'camvid_train_labels')
    masks_val = os.path.join(path, 'camvid_val_labels')
    masks_test = os.path.join(path, 'camvid_test_labels')

    files_images = os.listdir(images_train)
    for i in range(len(files_images)):
        filename = files_images[i]
        name, extension = os.path.splitext(filename)
        txt_temp = os.path.join(images_train, filename) + ' ' + \
                   os.path.join(masks_train, name + '_L' + extension) + '\n'
        train.write(txt_temp)
    files_images = os.listdir(images_val)
    for i in range(len(files_images)):
        filename = files_images[i]
        name, extension = os.path.splitext(filename)
        txt_temp = os.path.join(images_val, filename) + ' ' + \
                   os.path.join(masks_val, name + '_L' + extension) + '\n'
        val.write(txt_temp)
    files_images = os.listdir(images_test)
    for i in range(len(files_images)):
        filename = files_images[i]
        name, extension = os.path.splitext(filename)
        txt_temp = os.path.join(images_test, filename) + ' ' + \
                   os.path.join(masks_test, name + '_L' + extension) + '\n'
        test.write(txt_temp)

    train.close()
    val.close()
    test.close()

def colormap2label():
    COLORMAP = [[64, 128, 64], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0], [64, 0, 128], [64, 0, 192],
                [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192], [192, 0, 64], [128, 128, 64],
                [192, 0, 192], [128, 64, 64], [64, 192, 128], [64, 64, 0], [128, 64, 128], [128, 128, 192],
                [0, 0, 192], [192, 128, 128], [128, 128, 128], [64, 128, 192], [0, 0, 64], [0, 64, 64], [192, 64, 128],
                [128, 128, 0], [192, 128, 192], [64, 0, 64], [192, 192, 0], [0, 0, 0], [64, 192, 0]]
    # 32 classes
    # CLASS = ['name', 'Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CartLuggagePram', 'Child',
    #          'Column_Pole', 'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter',
    #          'OtherMoving', 'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky',
    #          'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel',
    #          'VegetationMisc', 'Void', 'Wall']

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
def mask2labelnonehot(mask, n_class=32, colormap2label=colormap2label()):
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
    COLORMAP = [[64, 128, 64], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0], [64, 0, 128], [64, 0, 192],
                [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192], [192, 0, 64], [128, 128, 64],
                [192, 0, 192], [128, 64, 64], [64, 192, 128], [64, 64, 0], [128, 64, 128], [128, 128, 192],
                [0, 0, 192], [192, 128, 128], [128, 128, 128], [64, 128, 192], [0, 0, 64], [0, 64, 64], [192, 64, 128],
                [128, 128, 0], [192, 128, 192], [64, 0, 64], [192, 192, 0], [0, 0, 0], [64, 192, 0]]
    COLORMAP = torch.Tensor(COLORMAP)
    mask = COLORMAP[label].permute([0, 3, 1, 2])
    return label, mask

def default_imgloader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
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
#         return img, onehot
        return img, label  # crossentropyloss: label_truth & onehot_prediction

    def __len__(self):
        return len(self.imagenmask)

if __name__ == '__main__':
    make_txt('camvid')
    transformer_image = transforms.Compose([transforms.Resize((360, 480)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])

    transformer_mask = transforms.Compose([transforms.Resize(size=(360, 480),
                                                             interpolation=transforms.InterpolationMode.NEAREST),
                                           transforms.ToTensor()])

    train_data = SegDataset(txt=os.path.join(os.getcwd(), 'camvid/train.txt'),
                            transform_image=transformer_image, transform_mask=transformer_mask)
    val_data = SegDataset(txt=os.path.join(os.getcwd(), 'camvid/val.txt'),
                          transform_image=transformer_image, transform_mask=transformer_mask)

    print('Num_of_train:', len(train_data))
    print('Num_of_val:', len(val_data))

    batch_size = 10
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    # optimization & loss function
    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCEWithLogitsLoss()

    # load pretrained vgg16_bn
    vgg16 = models.vgg16_bn(pretrained=True)
    vgg16.classifier = nn.Sequential()
    vgg16 = vgg16.state_dict()

    vgg16_rmfc = encoder()
    names = []
    values = []
    for key, value in vgg16_rmfc.state_dict().items():
        if "num_batches_tracked" in key:
            continue
        names.append(key)
    for key, value in vgg16.items():
        if "num_batches_tracked" in key:
            continue
        values.append(value)
    for key, value in zip(names, values):
        vgg16_rmfc.state_dict()[key] = value

    # freeze encoder
    for param in vgg16_rmfc.parameters():
        param.requires_grad = False

    # network
    n_class = 32
    device = torch.device('cuda:0')
    network = segnet(n_class=n_class, vgg16_rmfc=vgg16_rmfc)
    network = network.to(device)

    optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.9)

    # train
    epochs = 100
    for epoch in range(epochs):
        network.train()
        network.train()
        for data in iter(trainloader):
            # images, onehots = data
            images, labels = data
            # GPU
            # images, onehots = images.to(device), onehots.to(device)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # classifier
            outputs = network(images)
            loss = loss_function(outputs, labels)
            # 反向梯度传播
            loss.backward()
            # 用梯度做优化
            optimizer.step()
        network.eval()
        confusionmatrix = np.zeros([n_class, n_class])
        total = 0
        loss_total = 0
        with torch.no_grad():
            for data in iter(valloader):
                # images, onehots = data
                images, labels = data
                # GPU
                # images, onehots = images.to(device), onehots.to(device)
                images, labels = images.to(device), labels.to(device)
                outputs = network(images)
                loss_total += loss_function(outputs, labels)
                total += outputs.size(0)
                label_pred, mask_pred = onehot2labelnmask(outputs)
                # label_true, mask_true = onehot2labelnmask(onehots)
                label_true = labels
                label_pred, label_true = label_pred.cpu().numpy().reshape(-1), label_true.cpu().numpy().reshape(-1)
                # calculate confusion matrix
                confusionmatrix += confusion_matrix(label_true, label_pred, labels=list(range(0, n_class, 1)))
        metric = metrics(n_class=n_class, confusion_matrix=confusionmatrix)
        print('[val epoch: %d]loss: %.3f, pixel acc.: %.3f%%, mean acc.: %.3f%%, '
              'mean IU: %.3f%%, f.w. IU: %.3f%%, mean Dice: %.3f%%'
              % (epoch + 1, loss_total / total, metric.pixelaccuracy() * 100, metric.meanaccuracy() * 100,
                 metric.meaniu() * 100, metric.frequencyweightediu() * 100, metric.meandice() * 100))
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