from keras.models import load_model
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K

K.clear_session()

with open('vdata.pickle', 'rb') as handle:
    X_test, y_test = pickle.load(handle)

model = load_model('mod_50_0.3Drop.h5')

X_test = X_test[:, :, :, :, np.newaxis]
x = [X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3]]

test_preds = model.predict(x)
test_preds = np.reshape(test_preds, (test_preds.shape[0], 4, 4))
test_preds = np.argmax(test_preds, axis=2)
# test_preds = test_preds.squeeze()

num_examples = 50
idx = np.random.randint(0, X_test.shape[0], num_examples)
for i in idx:
    X_test1 = np.array(X_test[i])
    shape = X_test1[0].shape
    y_tmp = np.array(y_test[i]).astype(int)
    y_tmp_pred = np.array(test_preds[i]).astype(int)

    print(y_tmp)
    print(y_tmp_pred)

    X_new = [np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)]
    for ind, y in enumerate(y_tmp):
        X_new[y] = X_test1[ind]
    orig_img_1 = np.concatenate((X_new[0], X_new[1]), axis=1)
    orig_img_2 = np.concatenate((X_new[2], X_new[3]), axis=1)
    orig_img = np.concatenate((orig_img_1, orig_img_2), axis=0)
    X_new_pred = [np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)]
    for ind, y in enumerate(y_tmp_pred):
        X_new_pred[y] = X_test1[ind]
    pred_img_1 = np.concatenate((X_new_pred[0], X_new_pred[1]), axis=1)
    pred_img_2 = np.concatenate((X_new_pred[2], X_new_pred[3]), axis=1)
    pred_img = np.concatenate((pred_img_1, pred_img_2), axis=0)
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(orig_img.squeeze())
    fig.add_subplot(1, 2, 2)
    plt.imshow(pred_img.squeeze())
    fig.show()
