# -*- coding = utf-8 -*-
# @Time : 2022/6/5 16:34
# @Author : Luxlios
# @File : model_train.py
# @Software : PyCharm

import numpy as np
import SimpleITK as sitk  # Insight Toolkit
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import datetime
import logging
import sys
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from model.KiUNet_3d import KiUNet_3d
from model.UNet_3d import UNet_3d
from segmentation3d_metrics import dice_score


def make_txt(path, train_size=0.7, shuffle=True):
    # Divide the dataset into training set and test set and convert it to txt
    # --| path/(HGG/LGG)/brats_*
    #   --| VSD.Brain.XX.O.MR_Flair.*             # multimodal
    #       --| VSD.Brain.XX.O.MR_Flair.*.mha
    #   --| VSD.Brain.XX.O.MR_T1.*
    #       --| VSD.Brain.XX.O.MR_T1.*.mha
    #   --| VSD.Brain.XX.O.MR_T1c.*
    #       --| VSD.Brain.XX.O.MR_T1c.*.mha
    #   --| VSD.Brain.XX.O.MR_T2.*
    #       --| VSD.Brain.XX.O.MR_T2.*.mha
    #   --| VSD.Brain_3more.XX.O.OT.*             # label
    #       --| VSD.Brain_3more.XX.O.OT.*.mha
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

    hl = ['HGG', 'LGG']
    for types in hl:
        file = os.path.join(path, types)
        filenames = os.listdir(file)
        if shuffle:
            random.shuffle(filenames)
        else:
            pass
        for i in range(len(filenames)):
            if i < len(filenames) * train_size:
                filename = filenames[i]
                txt_temp = os.path.join(file, filename) + '\n'
                train.write(txt_temp)
            else:
                filename = filenames[i]
                txt_temp = os.path.join(file, filename) + '\n'
                test.write(txt_temp)
    train.close()
    test.close()


def normalize(img3d):
    '''
    :param img3d: 3d array
    :return: 3d array after normalize(z-score)
    '''
    img3d = (img3d - np.mean(img3d)) / np.std(img3d)
    return img3d


def img3d_loader(path):
    '''
    :param path: 3d format data
    :return: 3d array
    '''
    img3d = sitk.ReadImage(path)
    img3d = sitk.GetArrayFromImage(img3d)  # 155 * 240 * 240
    return img3d


def crop_center(img3d, size=160):
    '''
    :param img3d: 3d array
    :param size: expect size
    :return: 3d array after crop
    '''
    start = int((img3d.shape[1] - size) // 2)
    img3d = img3d[:, start:start+size, start:start+size]
    return img3d


def fill_depth(img3d, top_layer=2, bottom_layer=3):
    # fill depth to a number that is divisible by 2 multiple times(UpSample & MaxPool)
    '''
    :param img3d: 3d array
    :param top_layer: fill layer in top
    :param bottom_layer: fill layer in bottom
    :return: 3d array after fill
    '''
    none_layer = np.zeros([1, img3d.shape[1], img3d.shape[2]])
    for i in range(top_layer):
        img3d = np.concatenate([none_layer, img3d], axis=0)
    for i in range(bottom_layer):
        img3d = np.concatenate([img3d, none_layer], axis=0)
    return img3d


def mask2region(path, crop_size=160, fill_top=2, fill_bottom=3):
    # reference: https://www.smir.ch/BRATS/Start2015
    '''
    :param path: mask -- 3d format data
    :return: region array (4d)
    '''
    mask = img3d_loader(path)
    mask = fill_depth(crop_center(mask, size=crop_size), top_layer=fill_top, bottom_layer=fill_bottom)

    ct = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])  # Complete tumor
    tc = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])  # Tumor core
    et = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])  # Enhancing tumor
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k] == 0:
                    continue
                elif mask[i, j, k] == 1:
                    ct[i, j, k] = 1
                    tc[i, j, k] = 1
                elif mask[i, j, k] == 2:
                    ct[i, j, k] = 1
                elif mask[i, j, k] == 3:
                    ct[i, j, k] = 1
                    tc[i, j, k] = 1
                elif mask[i, j, k] == 4:
                    ct[i, j, k] = 1
                    tc[i, j, k] = 1
                    et[i, j, k] = 1
    region = np.array([ct, tc, et], dtype='float32')
    return region


def multimodal_fuse(path, crop_size=160, fill_top=2, fill_bottom=3):
    '''
    :param path: data path
    :return: multimodal fusion and 3 region fusion (array)
    '''
    # --| path
    #   --| VSD.Brain.XX.O.MR_Flair.*             # multimodal
    #       --| VSD.Brain.XX.O.MR_Flair.*.mha
    #   --| VSD.Brain.XX.O.MR_T1.*
    #       --| VSD.Brain.XX.O.MR_T1.*.mha
    #   --| VSD.Brain.XX.O.MR_T1c.*
    #       --| VSD.Brain.XX.O.MR_T1c.*.mha
    #   --| VSD.Brain.XX.O.MR_T2.*
    #       --| VSD.Brain.XX.O.MR_T2.*.mha
    #   --| VSD.Brain_3more.XX.O.OT.*             # label
    #       --| VSD.Brain_3more.XX.O.OT.*.mha
    # find address
    flair = glob.glob(os.path.join(path, 'VSD.Brain*.MR_Flair.*', '*.mha'))
    t1 = glob.glob(os.path.join(path, 'VSD.Brain*.MR_T1.*', '*.mha'))
    t1c = glob.glob(os.path.join(path, 'VSD.Brain*.MR_T1c.*', '*.mha'))
    t2 = glob.glob(os.path.join(path, 'VSD.Brain*.MR_T2.*', '*.mha'))
    mask = glob.glob(os.path.join(path, 'VSD.Brain*.OT.*', '*.mha'))

    # print(flair)

    # load data
    flair = normalize(img3d_loader(flair[0]))
    t1 = normalize(img3d_loader(t1[0]))
    t1c = normalize(img3d_loader(t1c[0]))
    t2 = normalize(img3d_loader(t2[0]))
    # print(flair.shape)

    # crop & fill
    flair = fill_depth(crop_center(flair, size=crop_size), top_layer=fill_top, bottom_layer=fill_bottom)
    t1 = fill_depth(crop_center(t1, size=crop_size), top_layer=fill_top, bottom_layer=fill_bottom)
    t1c = fill_depth(crop_center(t1c, size=crop_size), top_layer=fill_top, bottom_layer=fill_bottom)
    t2 = fill_depth(crop_center(t2, size=crop_size), top_layer=fill_top, bottom_layer=fill_bottom)

    # fusion
    multimodal = np.array([flair, t1, t1c, t2], dtype='float32')
    region = mask2region(mask[0], crop_size=crop_size, fill_top=fill_top, fill_bottom=fill_bottom)
    return multimodal, region


class SegDataset(Dataset):
    def __init__(self, txt, crop_size=160, fill_top=2, fill_bottom=3, loader=multimodal_fuse):
        super(SegDataset, self).__init__()
        txt_content = open(txt, 'r')
        filenames = []
        for line in txt_content:
            line = line.rstrip('\n')
            filenames.append(line)
        self.filenames = filenames
        self.loader = loader
        self.crop_size = crop_size
        self.fill_top = fill_top
        self.fill_bottom = fill_bottom

    def __getitem__(self, index):
        multimodal, region = self.loader(self.filenames[index], crop_size=self.crop_size,
                                         fill_top=self.fill_top, fill_bottom=self.fill_bottom)
        multimodal, region = torch.from_numpy(multimodal), torch.from_numpy(region)
        return multimodal, region

    def __len__(self):
        return len(self.filenames)


def logger_init(log_file_name='monitor', log_level=logging.DEBUG, log_dir='./logs/', only_file=False):
    # log path
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.datetime.now())[:10] + '_' +
                            str(datetime.datetime.now())[11:19].replace(':', '-') + '.log')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        # only output info to local file
        logging.basicConfig(filename=log_path, level=log_level, format=formatter, datefmt='%Y-%d-%m %H:%M:%S')
    else:
        # simultaneously output info to local file and terminal
        logging.basicConfig(level=log_level, format=formatter, datefmt='%Y-%d-%m %H:%M:%S',
                            handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])


if __name__ == '__main__':
    # multimodal, region = multimodal_fuse(r'G:\Desktop\gitnn\KiU-Net\BRATS2015_Training\HGG\brats_2013_pat0001_1')
    # print(multimodal.shape)
    # print(region.shape)
    path = 'BRATS2015_Training'
    make_txt(path, train_size=0.7, shuffle=True)

    train_data = SegDataset(txt=os.path.join(path, 'train.txt'), crop_size=240)
    test_data = SegDataset(txt=os.path.join(path, 'test.txt'), crop_size=240)
    # multimodal, region = train_data[0]
    # print(multimodal.shape)
    # print(region.shape)

    print('Num_of_train:', len(train_data))
    print('Num_of_test:', len(test_data))

    batch_size = 1
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # optimization & loss function
    loss_function = nn.BCEWithLogitsLoss()
    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # network = KiUNet_3d(in_channel=4, n_base=1, n_class=3).to(device)  # 4 modals, 3 regions
    network = UNet_3d(in_channel=4, n_class=3).to(device)
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    # train & test
    # tensorboard log
    # if os.path.exists('./tb-log'):
    #     shutil.rmtree('./tb-log')
    tb = SummaryWriter('./tb-log')
    logger_init(log_file_name='monitor', log_level=logging.INFO, log_dir='./logs')
    # checkpoint
    if not os.path.exists('./checkpoint'):
        os.mkdir('checkpoint')
    epoches = 100
    for epoch in range(epoches):
        network.train()
        loss_total = 0
        total = 0
        for data in tqdm(iterable=train_loader, desc='epoch {}'.format(epoch + 1), ncols=80, unit='imgs'):
            inputs, labels = data
            # GPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            loss_total += loss
            total += labels.size(0)
            # loss back propagation
            loss.backward()
            # optimized by sgd
            optimizer.step()
            time.sleep(0.1)
        loss_mean = loss_total / total
        # test dice
        network.eval()
        dice_total = 0
        ct_dice = 0
        tc_dice = 0
        et_dice = 0
        total = 0
        with torch.no_grad():
            for data in iter(test_loader):
                inputs, labels = data
                # GPU
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                # process
                outputs = torch.where(outputs > 0.5, 1, 0)
                # dice score
                ct_dice += dice_score(outputs[:, 0, :, :], labels[:, 0, :, :])
                tc_dice += dice_score(outputs[:, 1, :, :], labels[:, 1, :, :])
                et_dice += dice_score(outputs[:, 2, :, :], labels[:, 2, :, :])
                dice_total += dice_score(outputs, labels)
                total += labels.size(0)
        ct_dice = ct_dice / total
        tc_dice = tc_dice / total
        et_dice = et_dice / total
        dice_mean = dice_total / total
        # log
        tb.add_scalar(tag='loss',
                      scalar_value=loss_mean,
                      global_step=epoch)
        tb.add_scalar(tag='dice',
                      scalar_value=dice_mean,
                      global_step=epoch)
        logging.info(f'[epoch {epoch + 1: d}]loss:{loss_mean: .5f}, ct dice:{ct_dice: .5f}, '
                     f'tc dice:{ct_dice: .5f}, et dice:{et_dice: .5f}, dice:{dice_mean: .5f}')
        # save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            # save model
            save_dict = {
                'model': network.state_dict(),
                # 'optimizer': optimizer.state_dict()    # Adam optimizer need
            }
            torch.save(save_dict, './checkpoint/model_' + 'epoch' + str(epoch + 1) + '.pth.tar')
            '''
            # read model
            network = inception_v1(n_class=10, state='train')
            # optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.1)
            checkpoint = torch.load('./checkpoint/model.pth.tar')
            network.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            '''
        else:
            pass