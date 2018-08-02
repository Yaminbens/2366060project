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
    if tiles_per_dim == 4:
        size = [223, 223]

    Xd = []

    for file in files:
        img = cv2.imread(IM_DIR + file, cv2.IMREAD_GRAYSCALE)
        cover = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        Xd.append([file[0:-6]])
        cv2.imwrite(SAVE_DIR + file, cover)


    return np.array(Xd)


def data_prep():
    # update this number for 4X4 crop 2X2 or 5X5 crops.
    tiles_per_dim = 4

    X = data_transform(tiles_per_dim)

    with open('files_names' + '.pickle', 'wb') as handle:
        pickle.dump(X, handle)

def data_shrade():
    # update this number for 4X4 crop 2X2 or 5X5 crops.
    tiles_per_dim = 4

    augment()
    shrader(tiles_per_dim)



if __name__ == '__main__':
    # data_shrade()
    data_prep()
