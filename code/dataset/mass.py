from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import loadmat
import random
import torch


class SleepDataset(Dataset):

    def __init__(self, x_tensor, y_tensor, z_tensor):
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        return len(self.x)


def load_dataloader(data_train, data_test, label1_train, label1_test, label2_train, label2_test, batch_size, seed=2025):
    # Set random number generator for reproducible data loading
    g = torch.Generator()
    g.manual_seed(seed)

    # Set worker init function to ensure reproducible multi-process data loading
    train_iter = DataLoader(dataset=SleepDataset(data_train, label1_train, label2_train), batch_size=batch_size,
                            shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True,
                            generator=g, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    test_iter = DataLoader(dataset=SleepDataset(data_test, label1_test, label2_test), batch_size=batch_size,
                           shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True,
                           worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    return train_iter, test_iter


# def generate_transition(label, mode):
#     # Get the shape of label
#     n, length = label.shape
#     # Create a transition array with the same shape as label, initialized to 0
#     transition = np.zeros_like(label, dtype=int)
#     # Process according to different modes
#     if mode == 'left':
#         for i in range(n):
#             for j in range(length - 1):  # iterate over each label[i,j] to label[i,j+1]
#                 if label[i, j] != label[i, j + 1]:  # when labels differ
#                     transition[i, j] = 1
#     elif mode == 'right':
#         for i in range(n):
#             for j in range(length - 1):  # iterate over each label[i,j] to label[i,j+1]
#                 if label[i, j] != label[i, j + 1]:  # when labels differ
#                     transition[i, j + 1] = 1
#     elif mode == 'both':
#         for i in range(n):
#             for j in range(length - 1):  # iterate over each label[i,j] to label[i,j+1]
#                 if label[i, j] != label[i, j + 1]:  # when labels differ
#                     transition[i, j] = 1
#                     transition[i, j + 1] = 1
#     else:
#         raise ValueError("Mode must be one of 'left', 'right', or 'both'")
#     return transition

def generate_transition(label):
    # Vectorized implementation: an epoch is 0 (non-transition) when its label
    # matches both left and right neighbors, otherwise 1 (transition).
    # Boundary edges (first/last) treat the missing side as equal to itself,
    # which is exactly equivalent to the original element-wise logic.
    label = np.asarray(label)
    same_prev = np.ones_like(label, dtype=bool)
    same_next = np.ones_like(label, dtype=bool)
    same_prev[:, 1:] = label[:, 1:] == label[:, :-1]
    same_next[:, :-1] = label[:, :-1] == label[:, 1:]
    transition = (~(same_prev & same_next)).astype(int)
    return transition


def load_dataset(load_path, length, data_key, label_key):
    filePath = load_path
    datasets = loadmat(filePath)
    dataAll = np.array(datasets[data_key], dtype=np.float32)
    labelAll = np.array(datasets[label_key], dtype=np.int64).flatten()
    # [N, 30, 128], [N]
    n, len, dim = dataAll.shape
    last_full_batch_end_idx = n - n % length
    if n % length != 0:
        # Compute how many epochs to borrow from the front
        borrow_size = length - (n % length)
        # Create a new array combining all complete batches and the last merged batch
        dataAll = np.concatenate([dataAll[:last_full_batch_end_idx], np.concatenate(
            [dataAll[last_full_batch_end_idx - borrow_size:last_full_batch_end_idx],
             dataAll[last_full_batch_end_idx:]])]).reshape((n // length + 1), length, len, dim)
        labelAll = np.concatenate([labelAll[:last_full_batch_end_idx], np.concatenate(
            [labelAll[last_full_batch_end_idx - borrow_size:last_full_batch_end_idx],
             labelAll[last_full_batch_end_idx:]])]).reshape((n // length + 1), length)
    else:
        dataAll = dataAll.reshape(n // length, length, len, dim)
        labelAll = labelAll.reshape(n // length, length)
    transition_label = generate_transition(labelAll)
    return dataAll, labelAll, transition_label


def load_train_dataset(load_path, length, data_key, label_key, step):
    filePath = load_path
    datasets = loadmat(filePath)
    dataAll = np.array(datasets[data_key], dtype=np.float32)
    labelAll = np.array(datasets[label_key], dtype=np.int64).flatten()
    # [N, 30, 128], [N]
    n, len, dim = dataAll.shape
    start = 0
    data, label = [], []
    while start + length <= n:
        data.append(dataAll[start:start + length])
        label.append(labelAll[start:start + length])
        start += step
    data = np.array(data)
    label = np.array(label)
    transition_label = generate_transition(label)
    return data, label, transition_label


def mean_std(data):
    mean_acc = np.mean(data)
    std_acc = np.std(data)
    return mean_acc, std_acc

# file_dir = 'dataset/EDF20/mat/SC4001.mat'
# data, label, transition = load_dataset(file_dir, 32, 'PSD100', 'label')