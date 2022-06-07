# -*- coding = utf-8 -*-
# @Time : 2021/6/7 15:04
# @Author : Luxlios
# @File : model_test.py
# @Software : PyCharm

import torch
import os
import numpy as np
import glob
import shutil
import SimpleITK as sitk  # Insight Toolkit
from model_train import normalize, fill_depth, crop_center, img3d_loader
from model.UNet_3d import UNet_3d


def model_test(test_image, model_pth):
    # load model
    network = UNet_3d(in_channel=4, n_class=3)
    checkpoint = torch.load(model_pth)
    network.load_state_dict(checkpoint['model'])

    # load and fuse test image
    flair = glob.glob(os.path.join(test_image, 'VSD.Brain*.MR_Flair.*', '*.mha'))
    t1 = glob.glob(os.path.join(test_image, 'VSD.Brain*.MR_T1.*', '*.mha'))
    t1c = glob.glob(os.path.join(test_image, 'VSD.Brain*.MR_T1c.*', '*.mha'))
    t2 = glob.glob(os.path.join(test_image, 'VSD.Brain*.MR_T2.*', '*.mha'))

    # get some mha information to facilitate subsequent storage
    image = sitk.ReadImage(flair)
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    # normalize
    flair = normalize(img3d_loader(flair[0]))
    t1 = normalize(img3d_loader(t1[0]))
    t1c = normalize(img3d_loader(t1c[0]))
    t2 = normalize(img3d_loader(t2[0]))

    # crop & fill
    crop_size = 160
    fill_top = 2
    fill_bottom = 3
    flair = fill_depth(crop_center(flair, size=crop_size), top_layer=fill_top, bottom_layer=fill_bottom)
    t1 = fill_depth(crop_center(t1, size=crop_size), top_layer=fill_top, bottom_layer=fill_bottom)
    t1c = fill_depth(crop_center(t1c, size=crop_size), top_layer=fill_top, bottom_layer=fill_bottom)
    t2 = fill_depth(crop_center(t2, size=crop_size), top_layer=fill_top, bottom_layer=fill_bottom)

    # fusion
    multimodal = np.array([flair, t1, t1c, t2], dtype='float32')

    # test
    input = torch.from_numpy(multimodal)
    input = input.unsqueeze(0)
    output = network(input)
    output = torch.where(output > 0.5, 1, 0)
    output = output.squeeze(0).detach().numpy()
    output.dtype = np.int16

    ct = output[0, :, :, :]
    tc = output[1, :, :, :]
    et = output[2, :, :, :]

    # save dir
    if not os.path.exists(os.path.join(test_image, 'segmentation')):
        os.mkdir(os.path.join(test_image, 'segmentation'))
    else:
        shutil.rmtree(os.path.join(test_image, 'segmentation'))
        os.mkdir(os.path.join(test_image, 'segmentation'))

    ct = sitk.GetImageFromArray(ct)
    ct.SetSpacing(spacing)
    sitk.WriteImage(ct, os.path.join(test_image, 'segmentation', 'ct.mha'))

    tc = sitk.GetImageFromArray(tc)
    tc.SetSpacing(spacing)
    sitk.WriteImage(tc, os.path.join(test_image, 'segmentation', 'tc.mha'))

    et = sitk.GetImageFromArray(et)
    et.SetSpacing(spacing)
    sitk.WriteImage(et, os.path.join(test_image, 'segmentation', 'et.mha'))

if __name__ == '__main__':
    model_test(test_image='Testing/HGG_LGG/brats_2013_pat0103_1',
               model_pth='checkpoint/model_epoch200.pth.tar')
    # segmentation saved at 'Testing/HGG_LGG/brats_2013_pat0103_1/segmentation'
