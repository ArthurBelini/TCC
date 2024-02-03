import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from random import sample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

from odir_train_test_split import odir_train_test_split

odir_data_path = Path('../datasets/ODIR/codigo/256_preprocessed_classes')
X = []
y = []

max_imgs_per_label = 3000
for folder_name in os.listdir(odir_data_path):
    imgs_names = os.listdir(odir_data_path / folder_name)

    for img_name in sample(imgs_names, min(max_imgs_per_label, len(imgs_names))):
        X.append(img_name)
        y.append(folder_name)

classifiers = {'k-NN': KNeighborsClassifier(), 'SVM': SVC(), 'Random Forest': RandomForestClassifier(), 'Naive Bayes': GaussianNB()}
# classifiers = {'knn': KNeighborsClassifier()}

results_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'conf_matrix': []}
results = {classifier: deepcopy(results_metrics) for classifier in classifiers.values()}
iterations = 1
for i in range(iterations):
    print('Iteração:', i)

    X_train, X_test, y_train, y_test = odir_train_test_split(X, y, 0.2, odir_data_path)

    X_train  = [cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F).flatten() for img in X_train]
    X_test = [cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F).flatten() for img in X_test]

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
    print('Precisão:', np.mean(results[classifier]['prec']))
    print('Revocação:', np.mean(results[classifier]['rec']))
    print('F1 score:', np.mean(results[classifier]['f1']))

    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(sum(results[classifier]['conf_matrix']), display_labels=classifier.classes_)
    disp.plot()
    
plt.show()
