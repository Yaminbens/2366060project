import numpy as np
from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPool2D, Flatten, Concatenate, \
    Input, Reshape, Lambda
import keras
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras.models import Model
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

with open('data.pickle', 'rb') as handle:
    X, Y = pickle.load(handle)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

with open('vdata.pickle', 'wb') as handle:
    pickle.dump((X_test, y_test), handle)

image_shape = (X_train.shape[2], X_train.shape[3], 1)

X = None
Y = None


## TODO: when finished, train with all dataset!!

def sinkhorn_max(X, n_iter=10):
    xshape = (int(np.sqrt(len(X))), int(np.sqrt(len(X))))
    A = X.reshape(xshape)

    for i in range(n_iter):
        A = A / np.sum(A, axis=0)
        A = A / np.sum(A, axis=1)[:, np.newaxis]

    indices = []

    for i in range(int(np.sqrt(len(X)))):
        ind = np.argmax(A)
        indices.append(ind)
        mn = np.unravel_index(ind, (xshape))
        A[mn[0], :] = 0
        A[:, mn[1]] = 0

    Y01 = np.zeros(X.shape)
    Y01[indices] = 1
    return Y01


sinkhorn_on = False
kernel_size = (3, 3)
weight_decay = 0.005
dropout = 0.3

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
print(modelCNN.summary())


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
y = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay),
          kernel_initializer='glorot_uniform')(y)
y = BatchNormalization()(y)
final = Dense(16, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay),
              kernel_initializer='glorot_uniform')(y)
if sinkhorn_on:
    final = Reshape((4, 4))(final)
    final = Lambda(sinkhorn_max)(final)
    final = Reshape((16, 1))(final)


model = Model([x0, x1, x2, x3], final)

initial_lr = 0.01


# drop decay
def schedule(epoch):
    return initial_lr * (0.1 ** (epoch // 30))


lr_decay_drop_cb = LearningRateScheduler(schedule)
sgd = keras.optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


def data_generator(X_train, y_train, batch_size=128):
    while True:
        random_perm = np.array([np.random.permutation(4) for _ in range(batch_size)])
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        x_samples = X_train[idx, :]
        x_samples_rnd_perm = np.array([x_samples[i][random_perm[i]] for i in range(batch_size)])
        x_samples = x_samples_rnd_perm[:, :, :, :, np.newaxis]
        x = [x_samples[:, 0], x_samples[:, 1], x_samples[:, 2], x_samples[:, 3]]
        y = y_train[idx]
        y_rnd_perm = np.array([y[i][random_perm[i]] for i in range(batch_size)])
        y = keras.utils.to_categorical(y_rnd_perm, 4)
        y = np.reshape(y, (y.shape[0], 16))
        yield x, y

batch_size = 128

datagen = data_generator(X_train, y_train, batch_size=batch_size)
vdatagen = data_generator(X_test, y_test, batch_size=batch_size)

history = model.fit_generator(generator=datagen,
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              validation_data=vdatagen,
                              validation_steps=X_test.shape[0] // batch_size,
                              epochs=50,
                              callbacks=[lr_decay_drop_cb])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('mod_50_0.3Drop_lrdrop20ep.h5')

