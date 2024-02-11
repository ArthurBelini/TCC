import os
import shutil

import pandas as pd

from pathlib import Path

# data_path = Path.cwd().parent.parent.parent / 'preprocess' / '256_preprocessed_data'
data_path = Path.cwd().parent.parent.parent / 'preprocess' / '64_crop_data'
# data_path = Path('../data')
# out_data_path = Path('preprocessed_classes')
# out_data_path = Path('classes')
out_data_path = Path('64_crop_classes')
labels_csv_path = Path('labels.csv')

accepted_labels = ['N', 'D' ,'G' ,'C' ,'A' ,'H' ,'M' ,'O']

data = os.listdir(data_path)

if os.path.exists(out_data_path):
    shutil.rmtree(out_data_path)

os.mkdir(out_data_path)

for label in accepted_labels:
    out_data_file = out_data_path / label

    if not os.path.exists(out_data_file):
        os.mkdir(out_data_path / label)
    else:
        shutil.rmtree(out_data_path / label)
        os.mkdir(out_data_path / label)

# print(data)

with open(labels_csv_path) as labels_csv:
    global imgs_info

    imgs_info = pd.read_csv(labels_csv_path)
    imgs_info = [(labels[-1], labels[-3].strip('\'][').split(', ')) for labels in imgs_info.values.tolist()]

# print(data_labels)
    
for i, img_info in enumerate(imgs_info):
    print('Iteração:', i)

    img_name = img_info[0]

    for img_label in img_info[1]:
        if img_label not in accepted_labels:
            break

        src_file = data_path / img_name
        dst_file = out_data_path / img_label / img_name

        shutil.copyfile(data_path / img_name, out_data_path / img_label / img_name)
