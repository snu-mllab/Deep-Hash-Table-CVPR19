import numpy as np
import os

def jpg2png(str_):
    '''
    Args:
        str_ - .jpg
    '''
    return os.path.splitext(str_)[0]+'.png'

def rgb2gray(rgb_img):
    '''
    Args:
        rgb_img - Numpy 3D array
                [nrow, ncol ,3]
    Return:
        gray_img - Numpy 3D array
                [nrow, ncol ,1]
    '''
    gray_img = np.mean(rgb_img, axis=-1, keepdims=True)
    assert len(gray_img.shape)==3, 'Wrong operations'
    return gray_img

def gray2rgb(gray_img):
    '''
    Args:
        gray_img - Numpy 2D array
                [nrow, ncol]
    Return:
        rgb_img - Numpy 3D array
                [nrow, ncol ,3]

    '''
    w, h = gray_img.shape
    rgb_img = np.empty((w, h, 3), dtype=np.uint8)
    rgb_img[:, :, 0] = rgb_img[:, :, 1] = rgb_img[:, :, 2] = gray_img
    return rgb_img
