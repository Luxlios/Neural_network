# -*- coding = utf-8 -*-
# @Time : 2022/4/19 20:46
# @Author : Luxlios
# @File : segmentation_metrics.py
# @Software : PyCharm

import numpy as np

class metrics():
    def __init__(self, n_class, confusion_matrix):
        self.n_class = n_class
        self.confusion_matrix = confusion_matrix
    def pixelaccuracy(self):
        pixelaccuracy = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return pixelaccuracy

    def meanaccuracy(self):
        class_accuracy = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        meanaccuracy = class_accuracy.sum() / self.n_class
        return meanaccuracy

    def meaniu(self):
        # mean region intersection over union
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        iu = intersection / union
        meaniu = iu.sum() / self.n_class
        return meaniu

    def frequencyweightediu(self):
        frequency = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        iu = intersection / union
        frequencyweightediu = (frequency * iu).sum()
        return frequencyweightediu
