import scipy.io
import numpy as np

import scipy.io

# 加载 .mat 文件
mat = scipy.io.loadmat('./data1/MNIST_USPS.mat')

# 打印文件中的所有变量名
print(mat.keys())

# 步骤 1: 加载 .mat 文件
mat = scipy.io.loadmat('./data1/MNIST_USPS.mat')

# 提取变量 'X1', 'X2', 和 'Y'
X1 = mat['X1']
X2 = mat['X2']
Y = mat['Y']

# 保存为 .npz 文件
np.savez('data/MNIST_USPS.npz', X1=X1, X2=X2, Y=Y)





