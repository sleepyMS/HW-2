import sys
import os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
from fashion.fashion_mnist import load_fashion_mnist
from common.trainer import Trainer
from SimpleConvNet import SimpleConvNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=False)

max_epochs = 7

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=200, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=30,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 로그파일
with open('accuracy_log.txt', 'w') as log_file:
    log_file.write('Epoch\tTrain Accuracy\tTest Accuracy\n')
    for epoch, (train_acc, test_acc) in enumerate(zip(trainer.train_acc_list, trainer.test_acc_list)):
        log_file.write(f'{epoch}\t{train_acc}\t{test_acc}\n')

print("Train Accuracy: ", round(trainer.train_acc_list[-1], 4))
print("Test Accuracy: ", round(trainer.test_acc_list[-1], 4))