import cv2
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import datasets, layers, models
from pathlib import Path

from odir_train_test_split import odir_train_test_split

tf.get_logger().setLevel('ERROR')

def pre_process(img):
    # img = cv2.resize(img, (64, 64))
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

odir_data_path = Path('../datasets/ODIR/codigo/64_baseline_classes')
X = []
y = []
labels = {'D': 0, 'N': 1}

max_imgs_per_label = 1608
for folder_name in labels:
    imgs_path = odir_data_path / folder_name
    imgs_names = os.listdir(odir_data_path / folder_name)

    # for img_name in sample(imgs_names, min(max_imgs_per_label, len(imgs_names))):
    for img_name in imgs_names[:min(max_imgs_per_label, len(imgs_names))]:
        print(img_name)

        # img = cv2.imread(str(imgs_path / img_name))

        # img = pre_process(img)

        X.append(img_name)
        # X.append(img)
        y.append(folder_name)

X_train, X_test, y_train, y_test = odir_train_test_split(X, y, 0.2, odir_data_path)
X_train = np.array([pre_process(img) for img in X_train])
X_test = np.array([pre_process(img) for img in X_test])
y_train = np.array([labels[label] for label in y_train])
y_test = np.array([labels[label] for label in y_test])

print(y_train)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, 
                    validation_data=(X_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print(test_acc)
