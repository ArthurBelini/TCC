import cv2
import os
import tikzplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import albumentations as A

from keras import datasets, layers, models
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from keras.callbacks import Callback
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from collections import Counter
from random import sample
from models import model1, model2, model3, model4, model5, model6, model7, model8
from copy import deepcopy
from itertools import product

from odir_train_test_split import odir_train_test_split

tf.get_logger().setLevel('ERROR')

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

    # img = img.flatten()

    # img = np.array(img)

    return img

def augment(img):
    img = deepcopy(img)

    img = seq(image=img)

    # augmented = transform(image=img)
    # img = augmented['image']

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


class OnEpochEnd(Callback):
    def __init__(self, model, y_test, i):
        super(OnEpochEnd, self).__init__()
        self.model = model
        self.y_test = y_test
        self.i = i

    def on_epoch_end(self, epoch, logs=None):
        print('\nIteração:', i)

        y_pred = model.predict(X_test)
        y_pred = [list(pred).index(max(pred)) for pred in y_pred]

        val_accuracy = logs.get('val_accuracy')
        if val_accuracy > results['val_acc'][i]:
            accuracy = logs.get('accuracy')
            precision = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

            results['acc'][i] = accuracy
            results['val_acc'][i] = val_accuracy
            results['prec'][i] = precision
            results['rec'][i] = recall
            results['f1'][i] = f1
            results['conf_matrix'][i] = confusion_matrix(y_test, y_pred)
            results['epoch'][i] = epoch

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

# labels = ['N', 'D' ,'G' ,'C' ,'A' ,'H' ,'M' ,'O']
labels = ['N', 'D' ,'G' ,'C' ,'A', 'M']
# labels = ['N', 'D']
# labels = ['N', 'D', 'C']
res = 64
over_sample = True
augment_iterations = 1
augment_samples = 100
epochs = 10
batch_size = 64
iterations = 1
max_imgs_per_label = 300
color_channels = 1
# model_configs = [model1, model2, model3, model4, model5, model6, model7, model8]
# model_configs = [model3, model4, model5, model6, model7, model8]
# model_configs = [model2]
# model_configs = [model4]
# model_configs = [model8]
# model_configs = [model2, model8]
model_configs = [model2, model4, model8]
# over_configs = [RandomOverSampler, SMOTE]
over_configs = [SMOTE]

labels = dict(zip(labels, range(len(labels))))
odir_data_path = Path(f'../datasets/ODIR/codigo/{res}_classes')
X = []
y = []

for folder_name in labels:
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

# for model in [model3, model4, model5, model6, model7, model8]:
for cur_model, cur_over in product(model_configs, over_configs):
    results = {metric: [0] * iterations for metric in ['acc', 'val_acc', 'prec', 'rec', 'f1', 'conf_matrix', 'epoch']}
    max_val_acc_history = None
    for i in range(iterations):
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

        X_train = np.array([pre_process(img) for k, img in enumerate(X_train) if print('Pre-process treino:', k) or True])
        X_test = np.array([pre_process(img) for k, img in enumerate(X_test) if print('Pre-process teste:', k) or True])
        y_train = np.array([labels[label] for label in y_train])
        y_test = np.array([labels[label] for label in y_test])

        print(sorted(Counter(y_train).items()))

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        train_dataset.shuffle(buffer_size=len(y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        test_dataset.shuffle(buffer_size=len(y_train))

        model = cur_model((res, res, color_channels), len(labels))

        history = model.fit(train_dataset, epochs=epochs, 
                            validation_data=test_dataset,
                            # batch_size=batch_size,
                            callbacks=[OnEpochEnd(model, y_test, i)])
        
        if max_val_acc_history is None:
                max_val_acc_history = history

        elif history.history['val_accuracy'] > max_val_acc_history.history['val_accuracy']:
            max_val_acc_history = history

    plt.plot(max_val_acc_history.history['accuracy'], label='-'.join([cur_model.__name__, cur_over.__name__, 'accuracy']))
    plt.plot(max_val_acc_history.history['val_accuracy'], label='-'.join([cur_model.__name__, cur_over.__name__, 'val-accuracy']))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.ylim([1/len(labels), 1])
    # plt.legend(loc='upper left', ncol=1)
    plt.legend(loc='lower right', ncol=1)
    plt.title('Gráfico')
    plt.gcf().canvas.manager.set_window_title('Gráfico')

    print('Modelo:', cur_model.__name__)
    print('Over-sampling:', cur_over.__name__)
    print('Acurácia de validação:', np.mean(results["val_acc"]))
    print('Acurácia de validação max:', np.max(results["val_acc"]))
    print('Acurácia de validação std:', np.std(results["val_acc"]))
    print('Acurácia de treino:', np.mean(results["acc"]))

    print('Precisão:', np.mean(results["prec"]))
    print('Precisão std:', np.std([np.mean(x) for x in zip(*results["prec"])]))
    print('Precisão por rótulo:', [np.mean(x) for x in zip(*results["prec"])])

    print('Revocação:', np.mean(results["rec"]))
    print('Revocação std:', np.std([np.mean(x) for x in zip(*results["rec"])]))
    print('Revocação por rótulo:', [np.mean(x) for x in zip(*results["rec"])])

    print('F1 score:', np.mean(results["f1"]))
    print('F1 score std:', np.std([np.mean(x) for x in zip(*results["f1"])]))
    print('F1 score por rótulo:', [np.mean(x) for x in zip(*results["f1"])])

    print('Época de pico:', int(np.mean(results["epoch"])))
    print('Matriz de Confusão:\n', sum(results["conf_matrix"]) // iterations, sep='')
    
    with open('results.txt', 'a') as file:
        file.write(f'Modelo: {cur_model.__name__}\n')
        file.write(f'Over-sampling: {cur_over.__name__}\n')
        file.write(f'Acuracia de validacao: {np.mean(results["val_acc"])}\n')
        file.write(f'Acuracia de validacao max: {np.max(results["val_acc"])}\n')
        file.write(f'Acuracia de validacao Std: {np.std(results["val_acc"])}\n')
        file.write(f'Acuracia std: {np.std(results["acc"])}\n')
        file.write(f'Acuracia de treino: {np.mean(results["acc"])}\n')

        file.write(f'Precisao: {np.mean(results["prec"])}\n')
        file.write(f'Precisao std: {np.std([np.mean(x) for x in zip(*results["prec"])])}\n')
        file.write(f'Precisao por rotulo: {[np.mean(x) for x in zip(*results["prec"])]}\n')
        
        file.write(f'Revocacao: {np.mean(results["rec"])}\n')
        file.write(f'Revocacao std: {np.std([np.mean(x) for x in zip(*results["rec"])])}\n')
        file.write(f'Revocacao por rotulo: {[np.mean(x) for x in zip(*results["rec"])]}\n')

        file.write(f'F1 score: {np.mean(results["f1"])}\n')
        file.write(f'F1 score std: {np.std([np.mean(x) for x in zip(*results["f1"])])}\n')
        file.write(f'F1 score por rotulo: {[np.mean(x) for x in zip(*results["f1"])]}\n')

        file.write(f'Epoca de pico: {int(np.mean(results["epoch"]))}\n')
        file.write(f'Matriz de confusao:\n{sum(results["conf_matrix"]) // iterations}\n\n')

    # disp = ConfusionMatrixDisplay(sum(results['conf_matrix']) // iterations, display_labels=labels)
    # disp.plot()
    # plt.title('CNN')
    # plt.gcf().canvas.manager.set_window_title('CNN')

fig = plt.gcf()
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('plot.tex')
# tikzplotlib.save('_'.join([cur_model.__name__, cur_over.__name__]) + '.tex')

plt.show()
