import numpy as np
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
from keras.datasets import fashion_mnist
import keras.backend as K
from keras import regularizers
import keras
from sklearn.model_selection import train_test_split

def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


X_train, X_test = normalize(X_train, X_test)

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense( 512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay)))
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    model.summary()
    return model