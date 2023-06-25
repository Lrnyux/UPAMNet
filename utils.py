
import logging
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
import math
import torch.nn.functional as F
import torch
import lpips
# Generallly a refers to prediction and b refers to groundtruth


def calc_rmse(a, b, minmax=np.array([0,1])):

    a = a * (minmax[1] - minmax[0]) + minmax[0]
    b = b * (minmax[1] - minmax[0]) + minmax[0]

    return np.sqrt(np.mean(np.power(a - b, 2)))



def calc_psnr(a, b,minmax=np.array([0,1])):
    # img1 and img2 have range [0, 255]
    a = a * (minmax[1] - minmax[0]) + minmax[0]
    b = b * (minmax[1] - minmax[0]) + minmax[0]
    img1 = (a/np.max(a))*255
    img2 = (b/np.max(b))*255
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))



def calc_ssim(a, b,minmax=np.array([0,1])):
    a = a * (minmax[1] - minmax[0]) + minmax[0]
    b = b * (minmax[1] - minmax[0]) + minmax[0]
    img1 = (a / np.max(a)) * 255
    img2 = (b / np.max(b)) * 255
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_lpips(a, b, minmax=np.array([0,1]),loss=lpips.LPIPS(net='alex')):
    H,W = a.shape
    a = np.clip(a,-1,1)
    b = np.clip(b,-1,1)
    a = a.reshape(1,1,H,W)
    a = np.repeat(a,3,axis=1)
    b = b.reshape(1, 1, H, W)
    b = np.repeat(b, 3, axis=1)
    with torch.no_grad():
        return loss(torch.from_numpy(a).to(torch.float32),torch.from_numpy(b).to(torch.float32))


class CustomFormatter(logging.Formatter):
    DATE = '\033[94m'
    GREEN = '\033[92m'
    WHITE = '\033[0m'
    WARNING = '\033[93m'
    RED = '\033[91m'

    def __init__(self):
        orig_fmt = "%(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        super().__init__(orig_fmt, datefmt)

    def format(self, record):
        color = self.WHITE
        if record.levelno == logging.INFO:
            color = self.GREEN
        if record.levelno == logging.WARN:
            color = self.WARNING
        if record.levelno == logging.ERROR:
            color = self.RED
        self._style._fmt = "{}%(asctime)s {}[%(levelname)s]{} {}: %(message)s".format(
            self.DATE, color, self.DATE, self.WHITE)
        return logging.Formatter.format(self, record)


class ConsoleLogger():
    def __init__(self, training_type, phase='train'):
        super().__init__()
        self._logger = logging.getLogger(training_type)
        self._logger.setLevel(logging.INFO)
        formatter = CustomFormatter()
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_log.setFormatter(formatter)
        self._logger.addHandler(console_log)
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.logfile_dir = os.path.join('experiments/', training_type, time_str)
        os.makedirs(self.logfile_dir)
        logfile = os.path.join(self.logfile_dir, f'{phase}.log')
        file_log = logging.FileHandler(logfile, mode='a')
        file_log.setLevel(logging.INFO)
        file_log.setFormatter(formatter)
        self._logger.addHandler(file_log)

    def info(self, *args, **kwargs):
        """info"""
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """warning"""
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """error"""
        self._logger.error(*args, **kwargs)
        exit(-1)

    def getLogFolder(self):
        return self.logfile_dir


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.count = 0
        self.sum = 0
        self.ave = 0

    def update(self, val, num=1):
        self.count = self.count + num
        self.val = val
        self.sum = self.sum + num * val
        self.ave = self.sum / self.count if self.count != 0 else 0.0

def save_results(gt,pred,down,save_root_prefix,minmax=[0,1],gray=1):
    image_pred = pred * (minmax[1] - minmax[0]) + minmax[0]
    image_pred[image_pred < 0] = 0
    # image_pred[image_pred > 255] = 255
    image_gt = gt * (minmax[1] - minmax[0]) + minmax[0]
    image_gt[image_gt<0] = 0
    image_down = down * (minmax[1] - minmax[0]) + minmax[0]
    image_down[image_down < 0] = 0
    # image_down[image_down > 255] = 255
    if gray == 1:

        image_pred = np.array(image_pred,dtype=np.uint8)
        image_pred = Image.fromarray(image_pred)
        image_pred.save(save_root_prefix+'_pred.png')



        image_gt = np.array(image_gt,dtype=np.uint8)
        image_gt = Image.fromarray(image_gt)
        image_gt.save(save_root_prefix + '_gt.png')


        image_down = np.array(image_down, dtype=np.uint8)
        image_down = Image.fromarray(image_down)
        image_down.save(save_root_prefix + '_bi.png')

    if gray == 0:
        image_pred = np.array(image_pred, dtype=np.float64)
        plt.imsave(save_root_prefix + '_pred.png',image_pred,cmap='hot')

        image_gt = np.array(image_gt, dtype=np.float64)
        plt.imsave(save_root_prefix + '_gt.png', image_gt, cmap='hot')

        image_down = np.array(image_down, dtype=np.float64)
        plt.imsave(save_root_prefix + '_bi.png',image_down,cmap='hot')

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
