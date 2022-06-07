# -*- coding = utf-8 -*-
# @Time : 2021/6/4 22:02
# @Author : Luxlios
# @File : segmentation3d_metrics.py
# @Software : PyCharm

import torch

def dice_score(outputs, labels, eps=0.001):
    '''
    :param outputs: prediction (tensor)
    :param labels: target (tensor)
    :param eps: prevent denominator from being 0
    :return: dice score
    '''
    intersection = torch.sum(outputs * labels)
    union = torch.sum(outputs) + torch.sum(labels)
    dice = 2 * intersection / (union + eps)
    return dice