import os
import cv2
from keras.models import load_model
import numpy as np


def predict(images):
    if len(images) == 4:
        size = [120, 120]
        model = load_model('model2.h5')
    elif len(images) == 16:
        size = [60, 60]
        model = load_model('test4.h5')
    elif len(images) == 25:
        size = [60, 60]
        model = load_model('model5.h5')

    mean = 130
    std = 80
    images = (np.array(images) - mean) / (std + 1e-7)

    x = []
    for i in range(images.shape[0]):
        x.append(cv2.resize(images[i], (size[0], size[1]), interpolation=cv2.INTER_AREA)[np.newaxis, :, :, np.newaxis])

    test_preds = model.predict(x)
    test_preds = np.argmax(test_preds, axis=2)
    labels = list(test_preds)
    return labels


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images)
    return Y


print(evaluate())
