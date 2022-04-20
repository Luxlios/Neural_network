# -*- coding = utf-8 -*-
# @Time : 2022/4/19 14:48
# @Author : Luxlios
# @File : model_train.py
# @Software : PyCharm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from FCN_8s import fcn8s
from sklearn.metrics import confusion_matrix
from segmentation_metrics import metrics

def voc_colormap2label():
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]
    # 21 classes
    # VOC_CLASS = [
    #     'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #     'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    # torch.long <-> index(index get)  torch.uint8 <-> mask(0 mask, 1 get)
    # tensor index -- torch.long, not the same as numpy
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    # 0xffffff(256 256 256)
    # R,G,B -> [0, 255]
    for i, colormap in enumerate(VOC_COLORMAP):
        # Hex -> Dec
        # index: color
        # value: label
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

# 3 channels RGB image -> n_class channels onehot matrix
def voc_mask2labelnonehot(mask, n_class=21, colormap2label=voc_colormap2label()):
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

def voc_onehot2labelnmask(onehot):
    # onehot: tensor, batch * channel=21 * height * width

    # onehot to label
    label = torch.max(onehot, 1)[1]

    # label to mask
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]
    VOC_COLORMAP = torch.Tensor(VOC_COLORMAP)
    mask = VOC_COLORMAP[label].permute([0, 3, 1, 2])
    return label, mask


def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class SegDataset(Dataset):
    def __init__(self, voc_address, datatype='train', transform_image=None, transform_mask=None, loader=default_loader, mask2onehot=voc_mask2labelnonehot):
        super(SegDataset, self).__init__()
        txt = os.path.join(voc_address, 'VOC2012', 'ImageSets', 'Segmentation', datatype + '.txt')
        txt_content = open(txt, 'r')
        imagenmask = []
        for line in txt_content:
            line = line.rstrip('\n')
            image_address = os.path.join(voc_address, 'VOC2012', 'JPEGImages', line + '.jpg')
            mask_address = os.path.join(voc_address, 'VOC2012', 'SegmentationClass', line + '.png')
            imagenmask.append([image_address, mask_address])

        self.imagenmask = imagenmask
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.loader = loader
        self.mask2onehot = mask2onehot

    def __getitem__(self, index):
        image, mask = self.imagenmask[index]
        img = self.loader(image)
        mas = self.loader(mask)
        if self.transform_image is not None:
            img = self.transform_image(img)
        if self.transform_mask is not None:
            mas = self.transform_mask(mas)
        label, onehot = self.mask2onehot(mas)
        return img, onehot

    def __len__(self):
        return len(self.imagenmask)


if __name__ == '__main__':
    voc_address = os.path.join(os.getcwd(), 'VOCdevkit')

    transformer_image = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])

    # For segmentation task transformer_mask, bi_linear interpolation will generate new colormap(unknown class)
    # nearest interpolation can solve this problem
    transformer_mask = transforms.Compose([transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.NEAREST),
                                           transforms.ToTensor()])

    train_data = SegDataset(voc_address=voc_address, datatype='train', transform_image=transformer_image, transform_mask=transformer_mask)
    val_data = SegDataset(voc_address=voc_address, datatype='val', transform_image=transformer_image, transform_mask=transformer_mask)
    print('Num_of_train:', len(train_data))
    print('Num_of_val:', len(val_data))

    batch_size = 20
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    # GPU
    device = torch.device("cuda:0")
    n_class = 21
    vgg16_rmfc = models.vgg16(pretrained=True)
    vgg16_rmfc.classifier = nn.Sequential()
    # for param in list(vgg16_rmfc.parameters()):
    #     param.requires_grad = False

    network = fcn8s(vgg16_rmfc=vgg16_rmfc, n_class=n_class).to(device)
    network.cuda()
    # network = fcn8s(vgg16_rmfc=vgg16_rmfc, n_class=n_class)

    # optimization & loss function
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, network.parameters()), lr=0.001, momentum=0.9,
                          weight_decay=5E-4)
    # train
    epochs = 100
    for epoch in range(epochs):
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
            for data in iter(valloader):
                images, onehots = data
                # GPU
                images, onehots = images.to(device), onehots.to(device)
                outputs = network(images)
                loss_total += loss_function(outputs, onehots)
                total += outputs.size(0)
                label_pred, mask_pred = voc_onehot2labelnmask(outputs)
                label_true, mask_true = voc_onehot2labelnmask(onehots)
                label_pred, label_true = label_pred.cpu().numpy().reshape(-1), label_true.cpu().numpy().reshape(-1)
                # calculate confusion matrix
                confusionmatrix += confusion_matrix(label_true, label_pred)
        metric = metrics(n_class=n_class, confusion_matrix=confusionmatrix)
        print('[val epoch: %d]loss: %.3f, pixel acc.: %.3f%%, mean acc.: %.3f%%, mean IU: %.3f%%, f.w. IU: %.3f%%'
              %(epoch+1, loss_total / total, metric.pixelaccuracy() * 100, metric.meanaccuracy() * 100,
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

