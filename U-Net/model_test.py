# -*- coding = utf-8 -*-
# @Time : 2022/4/10 16:41
# @Author : Luxlios
# @File : model_test.py
# @Software : PyCharm

import torch
from torchvision import transforms
from UNet import UNet
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def model_test(test_image, model_pth):
    network = UNet(n_class=1)
    checkpoint = torch.load(model_pth)
    network.load_state_dict(checkpoint['model'])
    transformer_image = transforms.Compose([transforms.Resize((388, 388)),
                                            transforms.Pad(padding=92, padding_mode='reflect'),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])
    transformer_1 = transforms.Compose([transforms.Resize((572, 572)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
    image = Image.open(test_image)
    image = image.convert('L')
    image_1 = transformer_1(image)
    image_temp = transformer_image(image).unsqueeze(0)
    mask = network(image_temp)
    mask = mask.squeeze(0)
    mask = mask.detach().numpy()
    mask = np.transpose(mask, (1, 2, 0))

    image_1 = np.transpose(image_1, (1, 2, 0))

    plt.figure()
    plt.subplot(121)
    plt.imshow(image_1, cmap='gray')
    plt.subplot(122)
    plt.imshow(mask, cmap='gray')

    plt.show()

if __name__ == '__main__':
    model_test(test_image='tgs-salt-identification-challenge/test/images/0a0cc52eca.png', model_pth='model_epoch40.pth.tar')
