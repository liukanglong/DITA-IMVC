from torch.utils.data import Dataset, Sampler
import numpy as np
import torch
import os, sys
import scipy.io
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse


class MultiviewDataset(Dataset):
    def __init__(self, num_views, data_list, labels):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_views):
            data.append(torch.tensor(self.data_list[i][idx].astype('float32')))
        return data, torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class NGs(Dataset):
    def __init__(self, path):
        main_dir = sys.path[0]
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'NGs.mat'))
        X = mat['X'].T
        X = X[0]
        X_list = []
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        X_list.append(X[2].astype('float32'))
        self.V1 = X_list[0]
        self.V2 = X_list[1]
        self.V3 = X_list[1]
        self.Y = scipy.io.loadmat(path + 'NGs.mat')['Y'].astype(np.int32).reshape(500,)

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(2000)
        x2 = self.V2[idx].reshape(2000)
        x3 = self.V3[idx].reshape(2000)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class synthetic3d(Dataset):
    def __init__(self, path):
        main_dir = sys.path[0]
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'synthetic3d.mat'))
        X = mat['X'].T
        X = X[0]
        X_list = []
        X_list.append(X[0].astype('float32'))  # 3
        X_list.append(X[1].astype('float32'))  # 3
        X_list.append(X[2].astype('float32'))  # 3
        self.V1 = X_list[0]
        self.V2 = X_list[1]
        self.V3 = X_list[2]
        self.Y = scipy.io.loadmat(path + 'synthetic3d.mat')['Y'].astype(np.int32).reshape(600, )

    def __len__(self):
        return 600

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(3)
        x2 = self.V2[idx].reshape(3)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Cora(Dataset):
    def __init__(self, path):
        main_dir = sys.path[0]
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Cora.mat'))
        X_list = []
        X_list.append(mat['coracontent'].astype('float32'))
        X_list.append(mat['coracites'].astype('float32'))
        self.V1 = X_list[0]
        self.V2 = X_list[1]
        self.Y = scipy.io.loadmat(path + 'Cora.mat')['y'].astype(np.int32).reshape(2708, )

    def __len__(self):
        return 2708

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(1433)
        x2 = self.V2[idx].reshape(2708)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        data_list = []
        data_list.append(dataset.x1.astype('float32'))
        data_list.append(dataset.x2.astype('float32'))
        Y = dataset.y
        Y = Y.flatten()
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST_USPS":
        dataset = MNIST_USPS('./data/')
        data_list = []
        data_list.append(dataset.V1.reshape(5000, 784).astype('float32'))
        data_list.append(dataset.V2.reshape(5000, 784).astype('float32'))
        Y = dataset.Y
        Y = Y.flatten()
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        data_list = []
        data_list.append(dataset.data2.astype('float32'))
        data_list.append(dataset.data3.astype('float32'))
        data_list.append(dataset.data3.astype('float32'))
        Y = dataset.labels
        Y = Y.flatten()
        dims = [5000, 4000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "NGs":
        dataset = NGs('./data/')
        data_list = []
        data_list.append(dataset.V1.astype('float32'))
        data_list.append(dataset.V2.astype('float32'))
        data_list.append(dataset.V3.astype('float32'))
        Y = dataset.Y
        Y = Y.flatten()
        dims = [2000, 2000, 2000]
        view = 3
        class_num = 5
        data_size = 500
    elif dataset == "synthetic3d":
        dataset = synthetic3d('./data/')
        data_list = []
        data_list.append(dataset.V1.astype('float32'))
        data_list.append(dataset.V2.astype('float32'))
        data_list.append(dataset.V3.astype('float32'))
        Y = dataset.Y
        Y = Y.flatten()
        dims = [3, 3, 3]
        view = 3
        class_num = 3
        data_size = 600
    elif dataset == "Cora":
        dataset = Cora('./data/')
        data_list = []
        data_list.append(dataset.V1.astype('float32'))
        data_list.append(dataset.V2.astype('float32'))
        Y = dataset.Y
        Y = Y.flatten()
        dims = [1433, 2708]
        view = 2
        class_num = 7
        data_size = 2708

    else:
        raise NotImplementedError
    return data_list, Y, dims, view, data_size, class_num

class RandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
