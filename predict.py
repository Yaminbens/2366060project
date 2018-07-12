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
y_test = y_test.astype(int)

test_preds = model.predict(x)

num_examples = 5
idx = np.random.randint(0, X_test.shape[0], num_examples)
for i in idx:
    orig_img_1 = np.concatenate((X_test[i, y_test[i, 0]], X_test[i, y_test[i, 1]]), axis=1)
    orig_img_2 = np.concatenate((X_test[i, y_test[i, 2]], X_test[i, y_test[i, 3]]), axis=1)
    orig_img = np.concatenate((orig_img_1, orig_img_2), axis=0)
    # pred_img_1 = np.concatenate((X_test[i, test_preds[i, 0]], X_test[i, test_preds[i, 1]]), axis=0)
    # pred_img_2 = np.concatenate((X_test[i, test_preds[i, 2]], X_test[i, test_preds[i, 3]]), axis=0)
    # pred_img = np.concatenate((pred_img_1, pred_img_2), axis=1)
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(orig_img.squeeze())
    fig.add_subplot(1, 2, 2)
    # plt.imshow(pred_img)
    fig.show()
