import numpy as np
import keras
from keras.callbacks import LearningRateScheduler
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import model


# normalize - samplewise for CNN and featurewis for fullyconnected
def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


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


# drop decay
def schedule(epoch):
    return initial_lr * (0.1 ** (epoch // 50))


tiles_per_dim = 4

with open('data' + str(tiles_per_dim) + '.pickle', 'rb') as handle:
    X, Y = pickle.load(handle)

X = np.array(X).astype('float32') / 255
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
X_train, X_test = normalize(X_train, X_test)

with open('vdata' + str(tiles_per_dim) + '.pickle', 'wb') as handle:
    pickle.dump((X_test, y_test), handle)

image_shape = (X_train.shape[2], X_train.shape[3], 1)

X = None
Y = None

## TODO: when finished, train with all dataset!!

sinkhorn_on = False
weight_decay = 0.005
dropout = 0.3
initial_lr = 0.01
epochs = 45
batch_size = 64

model = model(tiles_per_dim, image_shape, sinkhorn_on, weight_decay, dropout)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
print(model.summary())

datagen = data_generator(X_train, y_train, batch_size=batch_size)
vdatagen = data_generator(X_test, y_test, batch_size=batch_size)

history = model.fit_generator(generator=datagen,
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              validation_data=vdatagen,
                              validation_steps=X_test.shape[0] // batch_size,
                              epochs=epochs,
                              callbacks=[LearningRateScheduler(schedule)])

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
