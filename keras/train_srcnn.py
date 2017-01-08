from __future__ import print_function, division

from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
from load_image import load_image
import numpy as np
import tensorflow as tf

np.random.seed(1337)  # for reproducibility

batch_size = 50
nb_epoch = 20

(X_train, Y_train), (X_test, Y_test) = load_image()

if K.image_dim_ordering() == 'tf':
    X_train = X_train.transpose(0, 2, 3, 1)
    Y_train = Y_train.transpose(0, 2, 3, 1)
    X_test = X_test.transpose(0, 2, 3, 1)
    Y_test = Y_test.transpose(0, 2, 3, 1)

print("start training...")


def log10(x):
    """
    tensorflowにはlog10がないので自分で定義
    https://github.com/tensorflow/tensorflow/issues/1666
    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    ref: https://github.com/titu1994/Image-Super-Resolution/blob/master/models.py
    """
    # tensorflowオブジェクトを返さないといけないので以下ではNG
    # return -10. * np.log10(K.mean(K.square(y_pred - y_true)))
    return -10. * log10(K.mean(K.square(y_pred - y_true)))

model = Sequential()
# backendでtensorflowを使っているときはinput_shape=(rows, cols, channels)の順。
model.add(Convolution2D(64, 9, 9, activation='relu', border_mode='valid', name='level1', input_shape=(64, 64, 3)))
# padding=0のときはborder_modelはvalid
model.add(Convolution2D(32, 1, 1, activation='relu', border_mode='valid', name='level2'))
model.add(Convolution2D(3, 5, 5, activation='relu', border_mode='valid', name='output'))
model.summary()

model.compile(loss='mse',
              optimizer=RMSprop(),
              metrics=[PSNRLoss])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
