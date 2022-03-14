import torch
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        return image


class Pad(object):
    def __init__(self, crop_size):
        self.resize_size = crop_size

    def __call__(self, image):
        w, h = image.size
        if h > w:
            padh = 0
            padw = h - w
        else:
            padh = w - h
            padw = 0
        image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)

        image = image.resize((self.resize_size, self.resize_size), Image.BILINEAR)

        return image
