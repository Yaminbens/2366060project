from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPool2D, Flatten, Concatenate, \
    Input, Reshape, Lambda
from keras import regularizers
from keras.models import Model
import keras.backend as K

def sinkhorn_max(A, n_iter=5):
    for i in range(n_iter):
        A /= K.sum(A, axis=0, keepdims=True)
        A /= K.sum(A, axis=1, keepdims=True)
    return A


def model(tiles_per_dim, image_shape, sinkhorn_on, weight_decay, dropout):
    if tiles_per_dim == 2:
        modelCNN = model2(image_shape, weight_decay)
    else:
        modelCNN = model4(image_shape, weight_decay)

    print(modelCNN.summary())

    x_in = []
    x_out = []
    for ind in range(2 ** tiles_per_dim):
        x_in.append(Input(shape=image_shape))
        x_out.append(modelCNN(x_in[ind]))

    concatenated = Concatenate()(x_out)
    y = Dropout(dropout)(concatenated)
    y = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay),
              kernel_initializer='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Dropout(dropout)(y)
    final = Dense(4 ** tiles_per_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay),
                  kernel_initializer='glorot_uniform')(y)
    final = Reshape((2 ** tiles_per_dim, 2 ** tiles_per_dim))(final)
    if sinkhorn_on:
        final = Lambda(sinkhorn_max)(final)

    model = Model(x_in, final)
    return model


def model2(image_shape, weight_decay):
    img_input = Input(shape=image_shape)
    x = Conv2D(64, (10, 10), padding='same', activation='relu', input_shape=image_shape,
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)
    x = MaxPool2D()(x)
    x = Conv2D(128, (7, 7), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Conv2D(128, (4, 4), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Conv2D(256, (4, 4), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Conv2D(256, (4, 4), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    out = BatchNormalization()(x)
    modelCNN = Model(img_input, out)
    return modelCNN


def model4(image_shape, weight_decay):
    img_input = Input(shape=image_shape)
    x = Conv2D(64, (7, 7), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)
    x = MaxPool2D()(x)
    x = Conv2D(128, (4, 4), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Conv2D(256, (4, 4), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Conv2D(256, (4, 4), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    out = BatchNormalization()(x)
    modelCNN = Model(img_input, out)
    return modelCNN
