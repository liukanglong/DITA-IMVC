import numpy as np

# 加载 .npz 文件
data = np.load('./data1/MNIST_USPS.mat')

# 打印文件中的所有条目名
print(data.files)