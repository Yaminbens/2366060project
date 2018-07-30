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
import keras.backend as K

tiles_per_dim = 4

with open('data' + str(tiles_per_dim) + '.pickle', 'rb') as handle:
    X, Y = pickle.load(handle)

X = np.array(X).astype('float32') / 255

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)


# data normalization
# normalize - samplewise for CNN and featurewis for fullyconnected
def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


X_train, X_test = normalize(X_train, X_test)

with open('vdata.pickle', 'wb') as handle:
    pickle.dump((X_test, y_test), handle)

image_shape = (X_train.shape[2], X_train.shape[3], 1)

X = None
Y = None


## TODO: when finished, train with all dataset!!

def sinkhorn_max(A, n_iter=5):
    for i in range(n_iter):
        A /= K.sum(A, axis=0, keepdims=True)
        A /= K.sum(A, axis=1, keepdims=True)
    return A


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
final = Dense(4 ** tiles_per_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay),
              kernel_initializer='glorot_uniform')(y)
final = Reshape((2 ** tiles_per_dim, 2 ** tiles_per_dim))(final)
if sinkhorn_on:
    final = Lambda(sinkhorn_max)(final)

model = Model(x_in, final)

initial_lr = 0.01


# drop decay
def schedule(epoch):
    return initial_lr * (0.1 ** (epoch // 35))


lr_decay_drop_cb = LearningRateScheduler(schedule)
sgd = keras.optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


def data_generator(X_train, y_train, batch_size=128):
    while True:
        random_perm = np.array([np.random.permutation(2 ** tiles_per_dim) for _ in range(batch_size)])
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        x_samples = X_train[idx, :]
        x_samples_rnd_perm = np.array([x_samples[i][random_perm[i]] for i in range(batch_size)])
        x_samples = x_samples_rnd_perm[:, :, :, :, np.newaxis]
        x = [x_samples[:, i] for i in range(x_samples.shape[1])]
        y = y_train[idx]
        y_rnd_perm = np.array([y[i][random_perm[i]] for i in range(batch_size)])
        y = keras.utils.to_categorical(y_rnd_perm, 2 ** tiles_per_dim)
        # for x_show, y_show in zip(x_samples,y):
        #     orig_img_1 = np.concatenate((x_show[0], x_show[1]), axis=1)
        #     orig_img_2 = np.concatenate((x_show[2], x_show[3]), axis=1)
        #     orig_img = np.concatenate((orig_img_1, orig_img_2), axis=0)
        #     fig = plt.figure()
        #     plt.imshow(orig_img.squeeze())
        #     fig.show()
        #     print(y_show)
        yield x, y


batch_size = 64

datagen = data_generator(X_train, y_train, batch_size=batch_size)
vdatagen = data_generator(X_test, y_test, batch_size=batch_size)

history = model.fit_generator(generator=datagen,
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              validation_data=vdatagen,
                              validation_steps=X_test.shape[0] // batch_size,
                              epochs=70,
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

model.save('test' + str(tiles_per_dim) + '.h5')
