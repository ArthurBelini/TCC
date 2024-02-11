import numpy as np

import cv2

from pathlib import Path
from math import floor, ceil

data_path = Path('../datasets/ODIR/data')

def prop_resize(img, size, use_min=False):
    function = max

    if use_min:
        function = min

    res = img.shape[1] / img.shape[0]

    new_size_x = int(function(res*size, size))
    new_size_y = int(function((1/res)*size, size))

    new_size = (new_size_x, new_size_y)

    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def crop_resize(img, size):
    img = prop_resize(img, size, use_min=False)

    res = img.shape[1] / img.shape[0]

    new_size_x_start = new_size_y_start = 0
    new_size_x_end = new_size_y_end = size + 1

    if res > 1:
        x_pixels = (img.shape[1] - size) / 2 

        new_size_x_start = floor(x_pixels)
        new_size_x_end = img.shape[1] - ceil(x_pixels)

    elif res < 1:
        y_pixels = (img.shape[0] - size) / 2 

        new_size_y_start = floor(y_pixels)
        new_size_y_end = img.shape[0] - ceil(y_pixels)

    else:
        return img

    # print(new_size_x_start, new_size_x_end, new_size_y_start, new_size_y_end)

    img = img[new_size_y_start : new_size_y_end, new_size_x_start : new_size_x_end]

    # print(img.shape[:2])

    return img

def pad_resize(img, size):
    img = prop_resize(img, size, True)

    res = img.shape[1] / img.shape[0]

    y_top_size = y_bottom_size = x_left_size = x_right_size = 0

    if res > 1:
        y_pixels = abs(img.shape[0] - size) / 2 

        y_top_size = floor(y_pixels)
        y_bottom_size = ceil(y_pixels)

    elif res < 1:
        x_pixels = abs(img.shape[1] - size) / 2 

        x_left_size = floor(x_pixels)
        x_right_size = ceil(x_pixels)

    # print(y_top_size, y_bottom_size, x_left_size, x_right_size)

    return cv2.copyMakeBorder(img, y_top_size, y_bottom_size, x_left_size, x_right_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

if __name__ == '__main__':
    img = cv2.imread(str(data_path / '0_left.jpg'))

    # img = prop_resize(img, 64, use_min=False)
    # img = crop_resize(img, 64)
    # img = pad_resize(img, 64)

    # print(img.shape)

    cv2.imshow('exemplo', img)
    cv2.waitKey()
