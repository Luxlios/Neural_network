# -*- coding = utf-8 -*-
# @Time : 2022/4/20 16:41
# @Author : Luxlios
# @File : model_test.py
# @Software : PyCharm

import torch
from torchvision import transforms, models
import torch.nn as nn
from FCN_8s import fcn8s
from model_train import voc_onehot2labelnmask
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def model_test(test_image, model_pth):
    vgg16_rmfc = models.vgg16(pretrained=True)
    vgg16_rmfc.classifier = nn.Sequential()
    network = fcn8s(vgg16_rmfc=vgg16_rmfc, n_class=21)
    checkpoint = torch.load(model_pth)
    network.load_state_dict(checkpoint['model'])
    transformer_image = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])
    image = Image.open(test_image)
    image = image.convert('RGB')
    image = transformer_image(image)

    image_test = image.unsqueeze(0)
    onehot_test = network(image_test)
    label_test, mask_test = voc_onehot2labelnmask(onehot_test)

    mask_test = mask_test.squeeze(0).permute([1, 2, 0])
    mask_test = mask_test.detach().numpy()

    image = image.permute([1, 2, 0]).numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.title('test image')
    plt.subplot(122)
    plt.imshow(mask_test)
    plt.title('mask prediction')

    plt.show()

if __name__ == '__main__':
    model_test(test_image='./VOCdevkit/VOC2012/JPEGImages/2007_000123.jpg', model_pth='model_epoch100.pth.tar')
