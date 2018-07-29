import os
import cv2
import numpy as np
from PIL import Image
from shrader_public import shrader
import pickle
import matplotlib.pyplot as plt


def augment():
    IM_DIR = "project/images/"
    SAVE_DIR = "project/allAug/"
    # PIL.Image.FLIP_LEFT_RIGHT

    files = os.listdir(IM_DIR)

    # update this number for 4X4 crop 2X2 or 5X5 crops.
    tiles_per_dim = 2
    for file in files:
        with open(IM_DIR + file, 'r+b') as f:
            with Image.open(f) as image:
                new_im_x = image.transpose(Image.FLIP_LEFT_RIGHT)
                new_im_y = image.transpose(Image.FLIP_TOP_BOTTOM)
                new_im_xy = image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)

                new_im_x.save(SAVE_DIR + file[0:-5] + 'x' + file[-5:], image.format)
                new_im_y.save(SAVE_DIR + file[0:-5] + 'y' + file[-5:], image.format)
                new_im_xy.save(SAVE_DIR + file[0:-5] + 'xy' + file[-5:], image.format)


def data_transform():
    IM_DIR = "project/shraded/"
    SAVE_DIR = "project/shraded_samesize/"
    files = os.listdir(IM_DIR)

    # update this number for 4X4 crop 2X2 or 5X5 crops.
    tiles_per_dim = 2

    # Define size of each tile
    size = [120, 120]

    Xd = {}
    Yd = {}

    for file in files:
        img = cv2.imread(IM_DIR + file, cv2.IMREAD_GRAYSCALE)
        cover = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        if file[0:-5] not in Xd:
            Xd.update({file[0:-5]: []})
            Yd.update({file[0:-5]: []})
        Xd[file[0:-5]].append(np.array(cover))
        Yd[file[0:-5]].append(file[-5:-4])
        cv2.imwrite(SAVE_DIR + file, cover)

    X = []
    Y = []

    for pic in Xd:
        # print(pic)

        X_test = np.array(Xd[pic])
        y_tmp = np.array(Yd[pic]).astype(int)
        X_new = [0 for _ in range(2 ** tiles_per_dim)]
        for ind, y in enumerate(y_tmp):
            X_new[y] = X_test[ind]

        X.append(X_new)
        Y.append([i for i in range(2 ** tiles_per_dim)])

        # orig_img_1 = np.concatenate((X_new[0], X_new[1]), axis=1)
        # orig_img_2 = np.concatenate((X_new[2], X_new[3]), axis=1)
        # orig_img = np.concatenate((orig_img_1, orig_img_2), axis=0)
        # fig = plt.figure()
        # plt.imshow(orig_img.squeeze())
        # fig.show()

    return np.array(X), np.array(Y)


def data_prep():
    augment()
    shrader()
    X, Y = data_transform()

    with open('data.pickle', 'wb') as handle:
        pickle.dump((X, Y), handle)


if __name__ == '__main__':
    data_prep()
