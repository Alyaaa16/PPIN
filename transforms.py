import random
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance
from PIL.ImageFilter import BLUR
from typing import Tuple

def CenterGaussianHeatMap(self, keypoints, height, weight, variance):
    c_x = keypoints[0]
    c_y = keypoints[1]
    gaussian_map = np.zeros((height, weight))
    for x_p in range(weight):
        for y_p in range(height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    # normalize
    xmax = max(map(max, gaussian_map))
    xmin = min(map(min, gaussian_map))
    gaussian_map_nor = (gaussian_map - xmin) / (xmax - xmin)
    return gaussian_map_nor


def _putGaussianMaps(self, keypoints, crop_size_y, crop_size_x, sigma):
    """
    :param keypoints: (24,2)
    :param crop_size_y: int  512
    :param crop_size_x: int  512
    :param stride: int  1
    :param sigma: float   1e-
    :return:
    """
    all_keypoints = keypoints  # 4,2
    point_num = all_keypoints.shape[0]  # 4
    heatmaps_this_img = []
    for k in range(point_num):  # 0,1,2,3
        flag = ~np.isnan(all_keypoints[k, 0])
        # heatmap = self._putGaussianMap(all_keypoints[k],flag,crop_size_y,crop_size_x,stride,sigma)
        heatmap = CenterGaussianHeatMap(self,keypoints=all_keypoints[k], height=crop_size_y, weight=crop_size_x,
                                             variance=sigma)
        heatmap = heatmap[np.newaxis, ...]
        heatmaps_this_img.append(heatmap)
    heatmaps_this_img = np.concatenate(heatmaps_this_img, axis=0)  # (num_joint,crop_size_y/stride,crop_size_x/stride)
    return heatmaps_this_img

def scale_box(xmin: float, ymin: float, w: float, h: float, scale_ratio: Tuple[float, float]):
    """根据传入的h、w缩放因子scale_ratio，重新计算xmin，ymin，w，h"""
    s_h = h * scale_ratio[0]
    s_w = w * scale_ratio[1]
    xmin = xmin - (s_w - w) / 2.
    ymin = ymin - (s_h - h) / 2.
    return xmin, ymin, s_w, s_h

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, heatmaps, gt):
        for t in self.transforms:
            image, heatmaps ,gt = t(image, heatmaps, gt)
        return image, heatmaps, gt


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, heatmaps, gt):

        image = np.asarray(image)/1.0
        image = torch.tensor(image,dtype=float)
        heatmaps = torch.tensor(heatmaps,dtype=float)
        gt = np.array(gt)
        gt = torch.tensor(gt,dtype=float)
        return image, heatmaps,gt


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image ,heatmaps, gt):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, heatmaps, gt

class Brightness(object):
    def __init__(self, prob=0.5):
        self.prob = prob


    def __call__(self, image ,heatmaps, gt):
        if random.random()<self.prob:
            image = Image.fromarray(image)
            enh_bri = ImageEnhance.Brightness(image)
            new_img = enh_bri.enhance(1.5)
            image = np.asarray(new_img)
        return image, heatmaps, gt


class Blur(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, heatmaps, gt):
        if random.random() < self.prob:
            image = Image.fromarray(image)
            image.filter(BLUR)
            image = np.asarray(image)
        return image, heatmaps, gt

class RandomCrop(object):
    def __init__(self,prob=0.5, H=512, W=512):
        self.factor = random.uniform(1.25, 1.5)
        self.prob =prob
        self.H = H
        self.W = W
    def __call__(self, image, heatmaps, gt):
        if random.random() < self.prob:
            selected_kps =  gt
            # crop image
            if len(selected_kps)> 2:
                xmin, ymin = np.min(selected_kps, axis=0).tolist()
                xmax, ymax = np.max(selected_kps, axis=0).tolist()
                w = xmax - xmin
                h = ymax - ymin
                xmin, ymin, w, h = scale_box(xmin, ymin, w, h, (1.5, 1.5))

            xmax = xmin+w
            ymax = ymin+h
            #xmin,xmax,ymin,ymax =xmin,xmax,ymin,ymax * self.factor
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            img = np.asarray(image)[ymin:ymax, xmin:xmax]
            heatmaps = np.asarray(heatmaps)[:,ymin:ymax, xmin:xmax]
            points = gt - np.array([xmin ,ymin])
        return img,heatmaps,points

class ReservePixel(object):
    def __init__(self,factor=None):
        self.reserve_factor = 512
    def __call__(self, image, heatmaps, gts):
        if random.random()<0.5:
            image = np.asarray(image)
            image = self.reserve_factor-image
            image = Image.fromarray(image)
        return image, heatmaps, gts

class RandomHorizontalFlip(object):
    """随机水平翻转图像"""
    def __init__(self, prob=0.5,H=512 ,W= 512):
        self.prob = prob
        self.H = H
        self.W = W
        global gt_flip
    def __call__(self, image, heatmaps , gt):
        if random.random() < self.prob:
            
            height, width = self.H, self.W
            image = np.ascontiguousarray(np.flip(image, axis=[1]))
            heatmaps = np.ascontiguousarray(np.flip(heatmaps, axis=[1]))
            gt = gt *np.array([1])
            gt[:,[0]] = height-gt[:,[0]]
        return image, heatmaps, gt
