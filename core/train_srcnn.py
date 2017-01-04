# coding: utf-8
from load_image import load_image
import numpy as np
import matplotlib.pyplot as plt
from srcnn import SRCNN
from common.trainer import Trainer


(x_train, t_train), (x_test, t_test) = load_image()

print("start training...")

max_epochs = 1000

network = SRCNN(params={'f1': 9, 'f2': 1, 'f3': 5, 'n1': 64, 'n2': 32, 'channel': 3},
                weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=50,
                  optimizer='Adam', optimizer_param={'lr': 0.01},
                  evaluate_sample_num_per_epoch=50)
trainer.train()

# パラメータの保存
network.save_params("./result/srcnn_params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("PSNR")
plt.legend(loc='lower right')
plt.show()
