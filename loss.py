import numpy as np
from keras import objectives
from keras import backend as K


_EPSILON = K.epsilon()
tiles = 2
def _loss_tensor(X, y_pred, y_true, CL,CR, tiles=2):
    X = K.permute_dimensions(X,(1,0,2,3))

    Xcs = K.reshape(X,(X.shape[0],X.shape[1],X.shape[2]*X.shape[3])) #reshape x as n columns

    X_pred = K.batch_dot(K.permute_dimensions(Xcs,(0,2,1)), y_pred) #replace column according to pred
    X_pred = K.reshape(X_pred,(X.shape[0],X.shape[-2]*tiles,X.shape[-1]*tiles)) #reshape as matrix
    X_flip_x = K.reverse(X_pred, axes=2)
    X_flip_y = K.reverse(X_pred, axes=1)

    LX_y = K.permute_dimensions(K.dot(CL, X_pred),(1,0,2))
    RX_y = K.permute_dimensions(K.dot(CL, X_flip_y),(1,0,2))
    LX_x = K.dot(X_pred, CR)
    RX_x = K.dot(X_flip_x, CR)
    dissimilarity = K.sum(K.sum(K.square(LX_x-RX_x), axis=2), axis=1) + K.sum(K.sum(K.square(LX_y-RX_y), axis=2), axis=1)

    return  K.categorical_crossentropy(y_pred, y_true) + K.sum(dissimilarity)


def check_loss(_shape, tiles=2):

    if _shape == '3d':
        shapex = (4,1, 6, 7)
        shapey = (1,4,4)
    elif _shape == '4d':
        shapex = (4, 8, 6, 7)
        shapey = (8,4,4)

    npixels = 3
    CL = np.zeros((shapex[-2] * tiles, shapex[-2] * tiles))
    for i in range(shapex[-2] - npixels, shapex[-2] + npixels -1):
        CL[i, i] = 1
    CR = np.zeros((shapex[-1] * tiles, shapex[-1] * tiles))
    for i in range(shapex[-2] - npixels, shapex[-2] + npixels -1):
        CR[i, i] = 1

    X = np.random.random(shapex)
    y = np.random.random(shapey)
    g = np.random.random(shapey)

    out1 = K.eval(_loss_tensor(K.variable(X), K.variable(y),  K.variable(g), K.variable(CL), K.variable(CR)))
    # out2 = _loss_np(X, y, g)

    # assert out1.shape == out2.shape
    # assert out1.shape == shapey[:-1]
    # print(np.linalg.norm(out1))
    # print(np.linalg.norm(out2))
    # print(np.linalg.norm(out1-out2))


def test_loss():
    shape_list = [ '4d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')

if __name__ == '__main__':
    test_loss()