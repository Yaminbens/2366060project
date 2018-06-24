import os
import cv2
import numpy as np
from PIL import Image

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

    sfiles = os.listdir(SAVE_DIR)
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