import os
import cv2
import numpy as np
from PIL import Image
from shrader_public import shrader


def augment():
    IM_DIR = "project/images/"
    SAVE_DIR = "project/imagesAug/"
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
                new_im_9 = image.transpose(Image.ROTATE_90)
                new_im_18 = image.transpose(Image.ROTATE_180)
                new_im_27 = image.transpose(Image.ROTATE_270)


                new_im_x.save(SAVE_DIR + file[0:-5] + 'x' + file[-5:], image.format)
                new_im_y.save(SAVE_DIR + file[0:-5] + 'y' + file[-5:], image.format)
                new_im_xy.save(SAVE_DIR + file[0:-5] + 'xy' + file[-5:], image.format)
                new_im_9.save(SAVE_DIR + file[0:-5] + '9' + file[-5:], image.format)
                new_im_18.save(SAVE_DIR + file[0:-5] + '18' + file[-5:], image.format)
                new_im_27.save(SAVE_DIR + file[0:-5] + '27' + file[-5:], image.format)





def data_transform():
    IM_DIR = "project/shraded/"
    SAVE_DIR = "project/shraded_samesize/"
    files = os.listdir(IM_DIR)

    # update this number for 4X4 crop 2X2 or 5X5 crops.
    tiles_per_dim = 2

    smin = 1010000
    smax = 0

    mwidth = []
    mheight = []

    widths = {}
    heights = {}


    for f in files:
        im = cv2.imread(IM_DIR+f)
        height = im.shape[0]
        if height not in heights:
            heights.update({height: 1})
        else:
            heights[height] += 1
        width = im.shape[1]
        if width not in widths:
            widths.update({width: 1})
        else:
            widths[width] += 1
        mwidth.append(width)
        mheight.append(height)
        s = height*width
        if s>smax:
            smax = s
            fmax = f
        if s<smin:
            smin = s
            fmin = f

    # print(np.median(mwidth))
    # print(np.median(mheight))

    size = [int(np.median(mwidth)), int(np.median(mheight))]

    Xd = {}
    Yd = {}

    sfiles = os.listdir(IM_DIR)
    for file in sfiles:
        with open(IM_DIR+file, 'r+b') as f:
            with Image.open(f) as image:
                cover = image.resize(size)
                if file[0:-7] not in Xd:
                    Xd.update({file[0:-7]: []})
                    Yd.update({file[0:-7]: []})
                Xd[file[0:-7]].append(np.array(cover))
                Yd[file[0:-7]].append(file[-5:-4])
                cover.save(SAVE_DIR+file, image.format)

    X = []
    Y = []

    for pic in Xd:
       X.append(Xd[pic])
       Y.append(Yd[pic])

    return np.array(X), np.array(Y)

if __name__ == '__main__':
    augment()
    shrader()
    X, y = data_transform()