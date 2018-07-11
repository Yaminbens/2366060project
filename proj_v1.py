import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPool2D, Flatten, Concatenate, \
    Input, Reshape
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras.models import Model
import pickle

with open('data.pickle', 'rb') as handle:
    (X, Y) = pickle.load(handle)

Y = keras.utils.to_categorical(Y, 4)
Y = np.reshape(Y, (Y.shape[0], 16))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)


## TODO: when finished, train with all dataset!!

def sinkhorn(A, n_iter=4):
    """
    Sinkhorn iterations.

    :param A: (n_batches, d, d) tensor
    :param n_iter: Number of iterations.
    """
    for i in range(n_iter):
        A /= A.sum(dim=1, keepdim=True)
        A /= A.sum(dim=2, keepdim=True)
    return A


# drop decay
def schedule(epoch):
    return 0.1 * (0.5 ** (epoch // 50))


lr_decay_drop_cb = LearningRateScheduler(schedule)
sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

sinkhorn_on = False
kernel_size = (3, 3)
image_shape = (X_train[0].shape[1], X_train[0].shape[2], 1)
weight_decay = 0
dropout = 0

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

x0 = Input(shape=image_shape)
x1 = Input(shape=image_shape)
x2 = Input(shape=image_shape)
x3 = Input(shape=image_shape)

x0_out = modelCNN(x0)
x1_out = modelCNN(x1)
x2_out = modelCNN(x2)
x3_out = modelCNN(x3)

concatenated = Concatenate()([x0_out, x1_out, x2_out, x3_out])
y = Dropout(dropout)(concatenated)
y = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(y)
y = BatchNormalization()(y)
final = Dense(16, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay))(y)
if sinkhorn_on:
    final = Reshape((4, 4))(final)
    final = sinkhorn(final)
    final = Reshape((16, 1))(final)

model = Model([x0, x1, x2, x3], final)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


def data_generator(X_train, y_train, batch_size=128):
    while True:
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        x_samples = X_train[idx,:]
        x_samples = x_samples[:, :, :, :, np.newaxis]
        x = [x_samples[:,0], x_samples[:,1], x_samples[:,2], x_samples[:,3]]
        y = y_train[idx]
        yield x, y


datagen = data_generator(X_train, y_train, batch_size=128)
vdatagen = data_generator(X_test, y_test, batch_size=128)

model.fit_generator(generator=datagen,
                    steps_per_epoch=X_train.shape[0] // 128,
                    validation_data=vdatagen,
                    validation_steps=X_test.shape[0] // 128,
                    epochs=100)
model.save('mod.h5')
