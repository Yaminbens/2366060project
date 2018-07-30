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
    for file in files:
        with open(IM_DIR + file, 'r+b') as f:
            with Image.open(f) as image:
                new_im_x = image.transpose(Image.FLIP_LEFT_RIGHT)
                new_im_y = image.transpose(Image.FLIP_TOP_BOTTOM)
                new_im_xy = image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)

                new_im_x.save(SAVE_DIR + file[0:-6] + 'x' + file[-6:], image.format)
                new_im_y.save(SAVE_DIR + file[0:-6] + 'y' + file[-6:], image.format)
                new_im_xy.save(SAVE_DIR + file[0:-6] + 'xy' + file[-6:], image.format)


def data_transform(tiles_per_dim):
    IM_DIR = "project/shraded" + str(tiles_per_dim) + "/"
    SAVE_DIR = "project/shraded_samesize" + str(tiles_per_dim) + "/"
    files = os.listdir(IM_DIR)

    # Define size of each tile
    if tiles_per_dim == 2:
        size = [120, 120]
    else:
        size = [60, 60]

    Xd = {}
    Yd = {}

    for file in files:
        img = cv2.imread(IM_DIR + file, cv2.IMREAD_GRAYSCALE)
        cover = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        if file[0:-6] not in Xd:
            Xd.update({file[0:-6]: []})
            Yd.update({file[0:-6]: []})
        Xd[file[0:-6]].append(np.array(cover))
        Yd[file[0:-6]].append(file[-6:-4])
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

        X.append(np.array(X_new))
        Y.append([i for i in range(2 ** tiles_per_dim)])

        # orig_img_1 = np.concatenate((X_new[0], X_new[1]), axis=1)
        # orig_img_2 = np.concatenate((X_new[2], X_new[3]), axis=1)
        # orig_img = np.concatenate((orig_img_1, orig_img_2), axis=0)
        # fig = plt.figure()
        # plt.imshow(orig_img.squeeze())
        # fig.show()

    return np.array(X), np.array(Y)


def data_prep():
    # update this number for 4X4 crop 2X2 or 5X5 crops.
    tiles_per_dim = 4

    # augment()
    # shrader(tiles_per_dim)
    X, Y = data_transform(tiles_per_dim)

    with open('data' + str(tiles_per_dim) + '.pickle', 'wb') as handle:
        pickle.dump((X, Y), handle)


if __name__ == '__main__':
    data_prep()
