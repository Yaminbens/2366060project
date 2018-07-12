import numpy as np
from keras import objectives
from keras import backend as K
from keras
_EPSILON = K.epsilon()

def _loss_tensor(X, y_pred, y_true, tiles):
    X = X.transpose((1,0,2,3))
    # y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    Xcs = K.reshape(X,(X.shape[0],X.shape[1],X.shape[1]*X.shape[2])) #reshape x as n columns
    X_pred = K.dot(Xcs.T, y_pred) #replace column according to pred
    X_pred = X_pred.reshape(X.shape[0],X.shape[-2]*tiles,X.shape[-1]*tiles) #reshape as matrix
    X_flip = K.reverse(K.reverse(X_pred, axes=0), axes=1)
    CL = K.zeros(X.shape[-2] * tiles, X.shape[-2] * tiles)
    CL[int(X.shape[-2]),int(X.shape[-2])] = 1
    CL[int(X.shape[-2])+1, int(X.shape[-2])+1] = 1
    CR = K.zeros(X.shape[-1] * tiles, X.shape[-1] * tiles)
    CR[int(X.shape[-1]), int(X.shape[-1])] = 1
    CR[int(X.shape[-1]) + 1, int(X.shape[-1]) + 1] = 1

    LX = K.dot(CL, K.X_pred) + K.dot((X_pred, CR))
    RX = K.dot(CL, K.X_flip) + K.dot((X_flip, CR))
    dissimilarity = K.sum(K.sum(K.square(LX-RX), axis=1), axis=0)

    return  K.categorical_crossentropy(y_pred, y_true) + dissimilarity


def _loss_np(X, y_pred, y_true, tiles):
    X = X.transpose((1,0,2,3))
    # y_pred = np.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    Xcs = np.reshape(X, (X.shape[0], X.shape[1], X.shape[1] * X.shape[2]))  # reshape x as n columns
    X_pred = np.dot(Xcs.T, y_pred)  # replace column according to pred
    X_pred = X_pred.reshape(X.shape[0], X.shape[-2] * tiles, X.shape[-1] * tiles)  # reshape as matrix
    X_flip = np.reverse(np.reverse(X_pred, axes=0), axes=1)
    CL = np.zeros(X.shape[-2] * tiles, X.shape[-2] * tiles)
    CL[int(X.shape[-2]), int(X.shape[-2])] = 1
    CL[int(X.shape[-2]) + 1, int(X.shape[-2]) + 1] = 1
    CR = np.zeros(X.shape[-1] * tiles, X.shape[-1] * tiles)
    CR[int(X.shape[-1]), int(X.shape[-1])] = 1
    CR[int(X.shape[-1]) + 1, int(X.shape[-1]) + 1] = 1

    LX = np.dot(CL, np.X_pred) + np.dot((X_pred, CR))
    RX = np.dot(CL, np.X_flip) + np.dot((X_flip, CR))
    dissimilarity = np.sum(np.sum(np.square(LX - RX), axis=1), axis=0)

    return K.categorical_crossentropy(y_pred, y_true) + dissimilarity

def check_loss(_shape, tiles):

    if _shape == '3d':
        shape = (1 ,4, 6, 7)
    elif _shape == '4d':
        shape = (8, 4, 6, 7)


    X_a = np.random.random(shape)
    y_b = np.random.random(shape)

    out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
    out2 = _loss_np(y_a, y_b)

    assert out1.shape == out2.shape
    assert out1.shape == shape[:-1]
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))


def test_loss():
    shape_list = ['2d', '3d', '4d', '5d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')

if __name__ == '__main__':
    test_loss()