import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch


def build_dataset(mode, window_size):

    rainfall = pd.read_csv("./dataset/Rainy_season/rs_RainFall.csv", low_memory=False)
    # rainfall.drop(['Unnamed: 단위(mm)', 'Unnamed: 0', 'date', 'time'], axis=1, inplace=True)
    rainfall = rainfall['길곡강수량']
    # waterlevel = pd.read_csv("./dataset/WaterLevel.csv", low_memory=False)
    waterlevel = pd.read_csv("./dataset/Rainy_season/rs_WaterLevel.csv", low_memory=False)
    waterlevel.drop(['Unnamed: 0', 'date', 'time'], axis=1, inplace=True)
    data = pd.concat([rainfall, waterlevel], axis=1)

    target = pd.read_csv("./dataset/Rainy_season/Target.csv", low_memory=False)
    target = target['총유입량']

    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    x_train = data[0 : train_size]
    x_valid = data[train_size - window_size : train_size + val_size]
    x_test = data[train_size + val_size - window_size :]

    y_train = target[0: train_size]
    y_valid = target[train_size - window_size : train_size + val_size]
    y_test = target[train_size + val_size - window_size :]


    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    x_valid = torch.tensor(x_valid.values, dtype=torch.float32)
    x_test = torch.tensor(x_test.values, dtype=torch.float32)

    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    if mode == 'train':
        x_time_series = x_train
        y_time_series = y_train

    if mode == 'valid':
        x_time_series = x_valid
        y_time_series = y_valid

    if mode == 'test':
        x_time_series = x_test
        y_time_series = y_test


    dataX = []
    dataY = []
    for i in range(0, len(x_time_series)-window_size):
        _x = x_time_series[i:i+window_size, :]
        _y = y_time_series[i+window_size, [-1]]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    dataset= TensorDataset(dataX, dataY)

    return dataset, dataX, dataY

# trainX, trainY = build_dataset(mode='train', window_size = 5)
# validX, validY = build_dataset(mode='valid', window_size = 5)
# testX, testY = build_dataset(mode='test', window_size=5)

# trian_dataset = build_dataset(mode='train', window_size = 5)



# # 텐서 형태로 데이터 정의
# dataset = TensorDataset(trainX_tensor, trainY_tensor)

# # 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
# dataloader = DataLoader(dataset,
#                         batch_size=batch,
#                         shuffle=True,  
#                         drop_last=True)