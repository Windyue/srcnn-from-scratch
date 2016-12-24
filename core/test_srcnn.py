# coding: utf-8
import numpy as np
from srcnn import SRCNN
import skimage.io
from PIL import Image


"""
ピクルファイルとテスト用画像を読み込んで結果を画像出力
"""

target_epoch = 100
test_image_index = 0

pkl_file = "./result/params_epoch_" + "{0:06d}".format(target_epoch) + ".pkl"
network = SRCNN()

network.load_params(pkl_file)

test_img = np.array(Image.open("./dataset/input/{0:05d}.bmp".format(test_image_index)))
test_img = test_img.transpose(2, 0, 1).astype(np.float32)/255.
d = test_img.reshape((1, test_img.shape[0], test_img.shape[1], test_img.shape[2]))

y = network.predict(d)
z, loss = network.last_layer.forward(y, t=None)

img_convert = z.reshape((z.shape[1], z.shape[2], z.shape[3])).transpose(1, 2, 0)
img_convert[np.where(img_convert>1.0)] = 1.0
img_convert = (img_convert*255).astype(np.uint8)
skimage.io.imsave("up_convert_epoch_{0:06d}.png".format(target_epoch), img_convert)