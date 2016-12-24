# coding: utf-8
import sys, os
sys.path.append(os.curdir)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.functions import *


class SRCNN:
    def __init__(self, params={'f1':9, 'f2':1, 'f3':5, 'n1':64, 'n2':32, 'channel':3}, 
                 weight_init_std=0.01):
        stride = 1
        pad = 0
        filter1_size = params['f1']
        filter2_size = params['f2']
        filter3_size = params['f3']
        filter1_num = params['n1']
        filter2_num = params['n2']
        channel = params['channel']

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter1_num, channel, filter1_size, filter1_size)
        self.params['b1'] = np.zeros(filter1_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(filter2_num, filter1_num, filter2_size, filter2_size)
        self.params['b2'] = np.zeros(filter2_num)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(channel, filter2_num, filter3_size, filter3_size)
        self.params['b3'] = np.zeros(channel)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride, pad)
        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], stride, pad)
        self.layers['Relu2'] = Relu()
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], stride, pad)
        self.last_layer = ReluWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        """ SRCNNではPSNR(ピーク信号雑音比)で評価する
        信号値を[0,1]で規格化しているので、最大値は255じゃなくて1
        """
        y = self.predict(x)
        mse = mean_squared_error(y, t)
        max_intensity = 1.0
        return 10 * np.log10(max_intensity*max_intensity/mse)

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝搬法）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        y, loss = self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db

        return grads, loss
        
    def save_params(self, file_name="srcnn_params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="srcnn_params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Conv3']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]