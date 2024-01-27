import os
import cv2
import shutil

from itertools import product
from pathlib import Path
from random import shuffle

class PersonEyes:
    right_name = None
    left_name = None
    right_label = None
    left_label = None

    def __init__(self, img_info, img_label):
        if img_info[1] == 'right.jpg':
            self.right_name = img_info[0]
            self.right_label = img_label
        else:
            self.left_name = img_info[0]
            self.left_label = img_label

    def set_eye(self, img_info, img_label):
        if img_info[1] == 'right.jpg':
            self.right_name = img_info[0]
            self.right_label = img_label
        else:
            self.left_name = img_info[0]
            self.left_label = img_label

def train_test_split(X: list, y: list, test_size : float = 0.20, data_path : Path = None, train_out_path : Path = None, test_out_path : Path = None):
    persons_eyes = {}
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    possible_pairs =  [{label1, label2} for label1, label2 in product(set(y).union({None}), repeat=2)]
    division_train = {tuple(set) : 0 for set in possible_pairs}
    division_test = {tuple(set) : 0 for set in possible_pairs}

    for img_name in X:
        img_info = img_name.split('_')
        img_label = y[X.index(img_name)]

        if not img_info[0] in persons_eyes:
            persons_eyes[img_info[0]] = PersonEyes(img_info, img_label)
        else:
            persons_eyes[img_info[0]].set_eye(img_info, img_label)

    persons_eyes = list(persons_eyes.items())
    shuffle(persons_eyes)
    persons_eyes = dict(persons_eyes)

    for eyes in persons_eyes.values():
        label_pair = {eyes.right_label, eyes.left_label}
        label_pair = tuple(label_pair)

        train_test_choice = division_test[label_pair] / (division_test[label_pair] + division_train[label_pair] + 1e-10) >= test_size

        choice = None
        if train_test_choice:
            choice = X_train, y_train, division_train
        else:
            choice = X_test, y_test, division_test

        if eyes.right_name:
            choice[0].append(eyes.right_name + '_right.jpg')
            choice[1].append(eyes.right_label)

        if eyes.left_name:
            choice[0].append(eyes.left_name + '_left.jpg')
            choice[1].append(eyes.left_label)

        choice[2][label_pair] += 1

    if data_path:
        X_train_names = X_train.copy()
        for i, img_name in enumerate(X_train):
            print('Iteração de Treino:', i)

            X_train[i] = cv2.imread(str(data_path / y_train[i] / img_name))

        X_test_names = X_test.copy()
        for i, img_name in enumerate(X_test):
            print('Iteração de Teste:', i)

            X_test[i] = cv2.imread(str(data_path / y_test[i] / img_name))

        if train_out_path and test_out_path:
            if os.path.exists(train_out_path):
                shutil.rmtree(train_out_path)
            os.mkdir(train_out_path)

            if os.path.exists(test_out_path):
                shutil.rmtree(test_out_path)
            os.mkdir(test_out_path)

            for label in os.listdir(data_path):
                os.mkdir(train_out_path / label)
                os.mkdir(test_out_path / label)

            for img_name, img, label in zip(X_train_names, X_train, y_train):
                cv2.imwrite(str(train_out_path / label / img_name), img)

            for img_name, img, label in zip(X_test_names, X_test, y_test):
                cv2.imwrite(str(test_out_path / label / img_name), img)

            return

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    data_path = Path('classes')
    train_out_path = Path('train')
    test_out_path = Path('test')
    X = []
    y = []

    for folder_name in os.listdir(data_path):
        for img_name in os.listdir(data_path / folder_name)[:100]:
            X.append(img_name)
            y.append(folder_name)

    train_test_split(X, y, data_path=data_path, train_out_path=train_out_path, test_out_path=test_out_path)
