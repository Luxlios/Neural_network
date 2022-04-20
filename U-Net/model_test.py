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
from model_train import onehot2labelnmask

def model_test(test_image, model_pth):
    network = UNet(n_class=2)
    checkpoint = torch.load(model_pth)
    network.load_state_dict(checkpoint['model'])
    transformer_image = transforms.Compose([transforms.Resize((388, 388)),  # 388 + 92 + 92 =572
                                            transforms.Pad(padding=92, padding_mode='reflect'),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])
    transformer_show = transforms.Compose([transforms.Resize((572, 572)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
    image = Image.open(test_image)
    image = image.convert('L')

    image_show = transformer_show(image)

    image_test = transformer_image(image).unsqueeze(0)
    onehot_test = network(image_test)
    label_test, mask_test = onehot2labelnmask(onehot_test)

    mask_test = mask_test.squeeze(0).permute([1, 2, 0])
    mask_test = mask_test.detach().numpy()

    image_show = image_show.permute([1, 2, 0]).numpy()

    plt.figure()
    plt.subplot(121)
    plt.imshow(image_show)
    plt.title('test image')
    plt.subplot(122)
    plt.imshow(mask_test)
    plt.title('mask prediction')

    plt.show()

if __name__ == '__main__':
    model_test(test_image='./tgs-salt-identification-challenge/test/images/0a0cc52eca.png', model_pth='model_epoch100.pth.tar')
