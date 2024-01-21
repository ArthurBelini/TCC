import cv2
import re

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

def train_test_split(X: list, y: list, test_size : float = 0.22, data_path : Path = None):
    persons_eyes = {}
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    possible_pairs =  [{label1, label2} for label1, label2 in product(set(y).union({None}), repeat=2)]
    division_train = division_test = \
        {tuple(set) : 0 for set in possible_pairs}

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

        train_test_choice = division_train[label_pair] * test_size < division_test[label_pair]

        choice = None
        if train_test_choice:
            choice = X_train, y_train, division_train
        else:
            choice = X_test, y_test, division_test

        if not eyes.right_name is None:
            choice[0].append(eyes.right_name + '_right.jpg')
            choice[1].append(eyes.right_label)

        if not eyes.left_name is None:
            choice[0].append(eyes.left_name + '_left.jpg')
            choice[1].append(eyes.left_label)

        choice[2][label_pair] += 1

    if not data_path is None:
        for i, img_name in enumerate(X_train):
            print('Iteração de Treino:', i)

            X_train[i] = cv2.imread(str(data_path / y_train[i] / img_name))

        for i, img_name in enumerate(X_test):
            print('Iteração de Teste:', i)

            X_test[i] = cv2.imread(str(data_path / y_test[i] / img_name))

    return X_train, X_test, y_train, y_test
