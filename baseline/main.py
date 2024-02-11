import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from random import sample, choice
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

odir_data_path = Path('../datasets/ODIR/codigo/128_baseline_classes')
X = []
y = []

max_imgs_per_label = 1608
for folder_name in ['D', 'N']:
    imgs_path = odir_data_path / folder_name
    imgs_names = os.listdir(odir_data_path / folder_name)

    # for img_name in sample(imgs_names, min(max_imgs_per_label, len(imgs_names))):
    for img_name in imgs_names[:min(max_imgs_per_label, len(imgs_names))]:
        print(img_name)

        img = cv2.imread(str(imgs_path / img_name))

        # img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.equalizeHist(img)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # img = clahe.apply(img)

        img = img.flatten()

        X.append(img)
        y.append(folder_name)


# cv2.imwrite('equalizeHist.jpg', X[30])

# cv2.imshow('Exemplo', choice(X))
# cv2.imshow('Exemplo', X[0])
# cv2.waitKey()

classifiers = {'k-NN': KNeighborsClassifier(), 'SVM': SVC()}
# classifiers = {'knn': KNeighborsClassifier()}

results_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'conf_matrix': []}
results = {classifier: deepcopy(results_metrics) for classifier in classifiers.values()}
iterations = 1
for i in range(iterations):
    print('Iteração:', i)

    # X_train, X_test, y_train, y_test = odir_train_test_split(X, y, 0.2, odir_data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # X_train  = [cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F).flatten() for img in X_train]
    # X_test = [cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F).flatten() for img in X_test]

    for classifier_name in classifiers:
        classifier = classifiers[classifier_name]

        print('Classificador:', classifier_name)

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results[classifier]['acc'].append(accuracy)
        results[classifier]['prec'].append(precision)
        results[classifier]['rec'].append(recall)
        results[classifier]['f1'].append(f1)
        results[classifier]['conf_matrix'].append(confusion_matrix(y_test, y_pred))

for classifier_name in classifiers:
    classifier = classifiers[classifier_name]

    print()
    print(f'{classifier_name}:')
    print('Acurácia:', np.mean(results[classifier]['acc']))
    print('Acurácia Std:', np.std(results[classifier]['acc']))
    print('Precisão:', np.mean(results[classifier]['prec']))
    print('Revocação:', np.mean(results[classifier]['rec']))
    print('F1 score:', np.mean(results[classifier]['f1']))

    disp = ConfusionMatrixDisplay(sum(results[classifier]['conf_matrix']) // iterations, display_labels=classifier.classes_)
    disp.plot()
    plt.title(classifier_name)
    plt.gcf().canvas.manager.set_window_title(classifier_name)
    
plt.show()
