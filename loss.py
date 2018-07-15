import numpy as np
from keras import objectives
from keras import backend as K
import pickle
import matplotlib.pyplot as plt
import keras

_EPSILON = K.epsilon()
tiles = 2


def _loss_tensor(X, y_pred, y_true, CL, CR, tiles=2):
    X = K.permute_dimensions(X, (1, 0, 2, 3))

    Xcs = K.reshape(X, (X.shape[0], X.shape[1], X.shape[2] * X.shape[3]))  # reshape x as n columns

    X_pred = K.batch_dot(K.permute_dimensions(Xcs, (0, 2, 1)), y_pred)  # replace column according to pred
    X_pred = K.reshape(X_pred, (X.shape[0], X.shape[-2] * tiles, X.shape[-1] * tiles))  # reshape as matrix
    X_flip_x = K.reverse(X_pred, axes=2)
    X_flip_y = K.reverse(X_pred, axes=1)

    LX_y = K.permute_dimensions(K.dot(CL, X_pred), (1, 0, 2))
    RX_y = K.permute_dimensions(K.dot(CL, X_flip_y), (1, 0, 2))
    LX_x = K.dot(X_pred, CR)
    RX_x = K.dot(X_flip_x, CR)
    dissimilarity = K.mean(K.mean(K.square(LX_x - RX_x), axis=2), axis=1) + K.mean(K.mean(K.square(LX_y - RX_y), axis=2),
                                                                                axis=1)

    # return K.categorical_crossentropy(y_pred, y_true) + K.sum(dissimilarity)
    return K.mean(dissimilarity)


def check_loss(_shape, tiles=2):
    if _shape == '3d':
        shapex = (4, 1, 6, 7)
        shapey = (1, 4, 4)
    elif _shape == '4d':
        shapex = (4, 8, 6, 7)
        shapey = (8, 4, 4)

    npixels = 1
    CL = np.zeros((shapex[-2] * tiles, shapex[-2] * tiles))
    for i in range(shapex[-2] - npixels, shapex[-2] + npixels):
        CL[i, i] = 1
    CR = np.zeros((shapex[-1] * tiles, shapex[-1] * tiles))
    for i in range(shapex[-2] - npixels, shapex[-2] + npixels):
        CR[i, i] = 1

    X = np.random.random(shapex)
    y = np.random.random(shapey)
    g = np.random.random(shapey)

    out1 = K.eval(_loss_tensor(K.variable(X), K.variable(y), K.variable(g), K.variable(CL), K.variable(CR)))
    # out2 = _loss_np(X, y, g)

    # assert out1.shape == out2.shape
    # assert out1.shape == shapey[:-1]
    # print(np.linalg.norm(out1))
    # print(np.linalg.norm(out2))
    # print(np.linalg.norm(out1-out2))


def test_loss():
    shape_list = ['4d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')


if __name__ == '__main__':


    with open('vdata.pickle', 'rb') as handle:
        X_test, y_test = pickle.load(handle)


    shapex = X_test.shape
    shapey = y_test
    npixels = 1
    CL = np.zeros((shapex[-2] * tiles, shapex[-2] * tiles))
    for i in range(shapex[-2] - npixels, shapex[-2] + npixels):
        CL[i, i] = 1
    CR = np.zeros((shapex[-1] * tiles, shapex[-1] * tiles))
    for i in range(shapex[-2] - npixels, shapex[-2] + npixels):
        CR[i, i] = 1

    num_examples = 1
    idx = np.random.randint(0, X_test.shape[0], num_examples)
    for i in idx:
        X_tmp = X_test[i]
        y_tmp = keras.utils.to_categorical(y_test[i], 4)
        y_tmp2 = keras.utils.to_categorical((y_test[i].astype(int) - 1) % 4, 4)
        X_tmp = K.variable(X_tmp[:, np.newaxis, :, :])
        y_tmp = K.variable(y_tmp[np.newaxis, :, :])
        y_tmp2 = K.variable(y_tmp2[np.newaxis, :, :])
        loss1 = _loss_tensor(X_tmp, y_tmp2, y_tmp, K.variable(CL), K.variable(CR))
        loss2 = _loss_tensor(X_tmp, y_tmp, y_tmp, K.variable(CL), K.variable(CR))
        X_test1 = np.array(X_test[i])
        shape = X_test1[0].shape
        y_tmp = np.array(y_test[i]).astype(int)
        X_new = [np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)]
        for ind, y in enumerate(y_tmp):
            X_new[y] = X_test1[ind]
        orig_img_1 = np.concatenate((X_new[0], X_new[1]), axis=1)
        orig_img_2 = np.concatenate((X_new[2], X_new[3]), axis=1)
        orig_img = np.concatenate((orig_img_1, orig_img_2), axis=0)
        fig = plt.figure()
        plt.imshow(orig_img.squeeze())
        plt.title('Loss1: ' + str(K.eval(loss1)) + ' Loss2: ' + str(K.eval(loss2)))
        fig.show()

    row = orig_img[int(orig_img.shape[0] / 2), :] - orig_img[int(orig_img.shape[0] / 2) + 1, :]

    fig = plt.figure()

    plt.imshow(orig_img)
    fig.show()
    # X_rec[:,X_rec.shape[2]]
    # test_loss()
