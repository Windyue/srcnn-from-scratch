# coding: utf-8
import numpy as np
import skimage.io
from PIL import Image
from keras.models import load_model


"""
ピクルファイルとテスト用画像を読み込んで結果を画像出力
"""
from keras import backend as K
import tensorflow as tf

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

target_epoch = 999
test_image_index = 0

model_file = "./result/model.ep" + "{0:06d}".format(target_epoch) + ".h5"
model = load_model(model_file, custom_objects={'PSNRLoss': PSNRLoss})

test_img = np.array(Image.open("./dataset/input/{0:05d}.bmp".format(test_image_index)))
test_img = test_img.transpose(2, 0, 1).astype(np.float32) / 255.
d = test_img.reshape((1, test_img.shape[1], test_img.shape[2], test_img.shape[0]))

y = model.predict(d)

img_convert = y.reshape((y.shape[1], y.shape[2], y.shape[3]))
img_convert[np.where(img_convert > 1.0)] = 1.0
img_convert = (img_convert * 255).astype(np.uint8)
skimage.io.imsave("up_convert_epoch_{0:06d}.png".format(target_epoch), img_convert)
