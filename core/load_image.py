# coding: utf-8
import numpy as np
from PIL import Image
import os.path


def load_image():
    train_num = 300  # 学習用データ数
    test_num = 50  # 検証用データ数

    train_in_img = []
    train_out_img = []
    test_in_img = []
    test_out_img = []

    # 学習用
    for i in range(train_num):
        # 画像を一つ読み込む
        in_im = np.array(Image.open("./dataset/input/{0:05d}.bmp".format(i)))
        out_im = np.array(Image.open("./dataset/output/{0:05d}.bmp".format(i)))
        # カラーを想定して3次元で
        im_in = in_im.transpose(2, 0, 1).astype(np.float32) / 255.
        im_out = out_im.transpose(2, 0, 1).astype(np.float32) / 255.
        train_in_img.append(im_in)
        train_out_img.append(im_out)

    # 検証用
    for i in range(train_num, train_num + test_num):
        # 画像を一つ読み込む
        in_im = np.array(Image.open("./dataset/input/{0:05d}.bmp".format(i)))
        out_im = np.array(Image.open("./dataset/output/{0:05d}.bmp".format(i)))
        # カラーを想定して3次元で
        im_in = in_im.transpose(2, 0, 1).astype(np.float32) / 255.
        im_out = out_im.transpose(2, 0, 1).astype(np.float32) / 255.
        test_in_img.append(im_in)
        test_out_img.append(im_out)

    return (np.array(train_in_img), np.array(train_out_img)), (np.array(test_in_img), np.array(test_out_img))


def load_fsrcnn_image(dir_prefix=""):
    train_num = 300  # 学習用データ数
    test_num = 50  # 検証用データ数

    train_in_img = []
    train_out_img = []
    test_in_img = []
    test_out_img = []

    __input_dir = os.path.join(".\\dataset\\", dir_prefix + "input")
    __output_dir = os.path.join(".\\dataset\\", dir_prefix + "output")

    # 学習用
    for i in range(train_num):
        # 画像を一つ読み込む
        in_im = np.array(Image.open(os.path.join(__input_dir, "{0:05d}.bmp".format(i))))
        out_im = np.array(Image.open(os.path.join(__output_dir, "{0:05d}.bmp".format(i))))
        # カラーを想定して3次元で
        im_in = in_im.transpose(2, 0, 1).astype(np.float32) / 255.
        im_out = out_im.transpose(2, 0, 1).astype(np.float32) / 255.
        train_in_img.append(im_in)
        train_out_img.append(im_out)

    # 検証用
    for i in range(train_num, train_num + test_num):
        # 画像を一つ読み込む
        in_im = np.array(Image.open(os.path.join(__input_dir, "{0:05d}.bmp".format(i))))
        out_im = np.array(Image.open(os.path.join(__output_dir, "{0:05d}.bmp".format(i))))
        # カラーを想定して3次元で
        im_in = in_im.transpose(2, 0, 1).astype(np.float32) / 255.
        im_out = out_im.transpose(2, 0, 1).astype(np.float32) / 255.
        test_in_img.append(im_in)
        test_out_img.append(im_out)

    return (np.array(train_in_img), np.array(train_out_img)), (np.array(test_in_img), np.array(test_out_img))
