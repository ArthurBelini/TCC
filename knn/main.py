import os
import cv2

import numpy as np

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from train_test_split import train_test_split

data_path = Path('data')
X = []
y = []

for folder_name in os.listdir(data_path):
    for img_name in os.listdir(data_path / folder_name):
        X.append(img_name)
        y.append(folder_name)

print(len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2, data_path)

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)
