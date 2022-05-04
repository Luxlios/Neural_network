# -*- coding = utf-8 -*-
# @Time : 2022/5/4 16:40
# @Author : Luxlios
# @File : model_test.py
# @Software : PyCharm

import torch
from torchvision import transforms, models
import torch.nn as nn
from SegNet import segnet, encoder
from model_train import onehot2labelnmask
import matplotlib.pyplot as plt
from PIL import Image

def model_test(test_image, model_pth):
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
    network = segnet(n_class=n_class, vgg16_rmfc=vgg16_rmfc)

    checkpoint = torch.load(model_pth)
    network.load_state_dict(checkpoint['model'])
    transformer_image = transforms.Compose([transforms.Resize((360, 480)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)])
    image = Image.open(test_image)
    image = image.convert('RGB')
    image = transformer_image(image)

    image_test = image.unsqueeze(0)
    onehot_test = network(image_test)
    label_test, mask_test = onehot2labelnmask(onehot_test)

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
    model_test(test_image='./camvid/camvid_test/Seq05VD_f03240.png', model_pth='model_epoch10.pth.tar')