import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import albumentations as A

from copy import deepcopy
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from random import sample, choice
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from odir_train_test_split import odir_train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from random import sample
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

def pre_process(img):
    # img = cv2.resize(img, (64, 64))
    if color_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.equalizeHist(img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # img = clahe.apply(img)

    img = np.float32(img)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # img /= 255.0

    img = img.flatten()

    return img

seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # random crops
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # blur images with a sigma between 0 and 0.5
    iaa.ContrastNormalization((0.75, 1.5)),  # contrast normalization
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),  # add Gaussian noise
    iaa.Multiply((0.8, 1.2), per_channel=0.2),  # multiply pixel values
    iaa.Affine(rotate=(-45, 45))  # rotate images
])

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(width=100, height=100),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
    ], p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.Rotate(limit=(-45, 45), p=0.5)
])

def augment(img):
    img = deepcopy(img)

    img = seq(image=img)

    # augmented = transform(image=img)
    # img = augmented['image']

    return img

odir_data_path = Path('../datasets/ODIR/codigo/64_classes')
X = []
y = []

res = 64
over_sample = True
augment_iterations = 1
augment_samples = 100
max_imgs_per_label = 300
iterations = 1
color_channels = 1
for folder_name in ['N', 'D' ,'G' ,'C' ,'A', 'M']:
    imgs_path = odir_data_path / folder_name
    imgs_names = os.listdir(odir_data_path / folder_name)

    for img_name in sample(imgs_names, min(max_imgs_per_label, len(imgs_names))):
    # for img_name in imgs_names[:min(max_imgs_per_label, len(imgs_names))]:
        print(img_name)

        # img = cv2.imread(str(imgs_path / img_name))

        # img = pre_process(img)

        X.append(img_name)
        # X.append(img)
        y.append(folder_name)


# cv2.imwrite('equalizeHist.jpg', X[30])

# cv2.imshow('Exemplo', choice(X))
# cv2.imshow('Exemplo', X[0])
# cv2.waitKey()

classifiers = {'k-NN': KNeighborsClassifier(), 'SVM': SVC(verbose=True), 'Random Forest': RandomForestClassifier(verbose=1), 'Naive Bayes': GaussianNB()}
# classifiers = {'knn': KNeighborsClassifier()}

results_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'conf_matrix': []}
results = {classifier: deepcopy(results_metrics) for classifier in classifiers.values()}
for i in range(iterations):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, X_test, y_train, y_test = odir_train_test_split(X, y, 0.3, odir_data_path)

    if over_sample:
        X_train = X_train.reshape(X_train.shape[0], -1)
        # X_train, y_train = RandomOverSampler().fit_resample(X_train, y_train)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        print(X_train.shape)
        X_train = X_train.reshape(-1, res, res, 3)

    X_train_original = deepcopy(X_train)
    y_train_original = deepcopy(y_train)
    for _ in range(augment_iterations):
        X_train_test = zip(X_train_original, y_train_original)
        X_train_test = sample(list(X_train_test), augment_samples)
        X_train_test_augmented = [(augment(img), label) for img, label in X_train_test]

        X_train_augmented, y_train_augmented = zip(*X_train_test_augmented)

        X_train = np.concatenate((X_train, X_train_augmented))
        y_train = np.concatenate((y_train, y_train_augmented)) 

    print(X_train.shape)
    X_train = [pre_process(img) for img in X_train]
    X_test = [pre_process(img) for img in X_test]

    print(sorted(Counter(y_train).items()))

    print('Iteração:', i)
    for classifier_name in classifiers:
        classifier = classifiers[classifier_name]

        print('Classificador:', classifier_name)

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

        results[classifier]['acc'].append(accuracy)
        results[classifier]['prec'].append(precision)
        results[classifier]['rec'].append(recall)
        results[classifier]['f1'].append(f1)
        results[classifier]['conf_matrix'].append(confusion_matrix(y_test, y_pred))

for classifier_name in classifiers:
    classifier = classifiers[classifier_name]

    print()
    print(f'{classifier_name}:')
    print('Acurácia de validação:', np.mean(results[classifier]["acc"]))
    print('Acurácia de validação max:', np.max(results[classifier]["acc"]))
    print('Acurácia de validação Std:', np.std(results[classifier]["acc"]))
    print('Precisão:', np.mean(results[classifier]["prec"]), np.std([np.mean(x) for x in zip(*results[classifier]["prec"])]), [np.mean(x) for x in zip(*results[classifier]["prec"])])
    print('Revocação:', np.mean(results[classifier]["rec"]), np.std([np.mean(x) for x in zip(*results[classifier]["rec"])]), [np.mean(x) for x in zip(*results[classifier]["rec"])])
    print('F1 score:', np.mean(results[classifier]["f1"]), np.std([np.mean(x) for x in zip(*results[classifier]["f1"])]), [np.mean(x) for x in zip(*results[classifier]["f1"])])
    print('Matriz de Confusão:\n', sum(results[classifier]["conf_matrix"]) // iterations, sep='')

    with open('results2.txt', 'a') as file:
        file.write(f'{classifier_name}:\n')
        file.write(f'Acuracia de validacao: {np.mean(results[classifier]["acc"])}\n')
        file.write(f'Acuracia de validacao max: {np.max(results[classifier]["acc"])}\n')
        file.write(f'Acuracia de validacao Std: {np.std(results[classifier]["acc"])}\n')
        file.write(f'Acuracia std: {np.std(results[classifier]["acc"])}\n')
        file.write(f'Acuracia de treino: {np.mean(results[classifier]["acc"])}\n')
        file.write(f'Precisao: {np.mean(results[classifier]["prec"])} {np.std([np.mean(x) for x in zip(*results[classifier]["prec"])])} {[np.mean(x) for x in zip(*results[classifier]["prec"])]}\n')
        file.write(f'Revocacao: {np.mean(results[classifier]["rec"])} {np.std([np.mean(x) for x in zip(*results[classifier]["rec"])])} {[np.mean(x) for x in zip(*results[classifier]["rec"])]}\n')
        file.write(f'F1 score: {np.mean(results[classifier]["f1"])} {np.std([np.mean(x) for x in zip(*results[classifier]["f1"])])} {[np.mean(x) for x in zip(*results[classifier]["f1"])]}\n')
        file.write(f'Matriz de confusao:\n{sum(results[classifier]["conf_matrix"]) // iterations}\n\n')

    # disp = ConfusionMatrixDisplay(sum(results[classifier]['conf_matrix']) // iterations, display_labels=classifier.classes_)
    # disp.plot()
    # plt.title(classifier_name)
    # plt.gcf().canvas.manager.set_window_title(classifier_name)
    
# plt.show()
