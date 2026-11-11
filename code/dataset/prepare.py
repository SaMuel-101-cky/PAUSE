# python3 PrepareDataset.py
from sklearn.model_selection import KFold
from scipy.io import savemat, loadmat
from dataset.mass import load_train_dataset
import numpy as np
import os
from dataset.stft import STFT

data_dir = 'dataset/EDF20/mat/'
file_list = os.listdir(data_dir)
arrays = [load_train_dataset(data_dir + file_name, 32, 'PSD100', 'label', 3)[0] for file_name in file_list]
data_all = np.concatenate(arrays, axis=0)
print(data_all.shape)
kf = KFold(10, shuffle=True, random_state=2024)
fold_num = 1
for train_index, test_index in kf.split(data_all):
    # split train and test data
    savemat('dataset/EDF20/fold/Fold{}.mat'.format(fold_num), {'index_train': train_index, 'index_test': test_index})
    fold_num += 1
