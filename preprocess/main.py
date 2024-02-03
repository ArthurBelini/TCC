import cv2
import os
import shutil

from random import sample
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

data_path = Path('../datasets/ODIR/data')
# out_data_path = Path('preprocessed_data')
out_data_path = Path('256_preprocessed_data')

average_imgs_size = (0, 0)

if out_data_path.exists():
    shutil.rmtree(out_data_path)

os.mkdir(out_data_path)

imgs_names = os.listdir(data_path)[:]
for i, img_name in enumerate(imgs_names):
    print('Iteração:', i)

    img = cv2.imread(str(data_path / img_name))

    # print(img.shape[:2])
    average_imgs_size = tuple(sum(dim_sizes) for dim_sizes in zip(average_imgs_size, img.shape[:2]))

    # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)

    # img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # (img * 255).astype('uint8')
    
    cv2.imwrite(str(out_data_path / img_name), img)

# samples_size = 3
# samples = sample(os.listdir(out_data_path), samples_size)
# for i in range(samples_size):
#     img_name = samples[i]

#     img = cv2.imread(str(out_data_path / samples[i]))
    
#     cv2.imshow(img_name, img)

# cv2.waitKey()

average_imgs_size = tuple(dim_size // len(imgs_names) for dim_size in average_imgs_size)

print('Average Images Size:', average_imgs_size)
