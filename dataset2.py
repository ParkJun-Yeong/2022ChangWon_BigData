"""
강수량, 상류지 수위 등을 학습하는 모델을 위한 데이터 로더
"""

from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


class Data(Dataset):
    def __init__(self, mode, window_size, input_size):
        super(Data, self).__init__()

        self.mode = mode
        self.window_size = window_size
        self.input_size = input_size
        self.interval = 3

        rainfall = pd.read_csv("./dataset/RainFall.csv", low_memory=False)

        # 아래 2줄 rainfall 을 피처로 추가하는 코드
        rainfall = pd.read_csv("./dataset/scaled/RainFall_scaled.csv", low_memory=False)
        rainfall.drop(['Unnamed: 단위(mm)', 'Unnamed: 0', 'date', 'time'], axis=1, inplace=True)


        # waterlevel = pd.read_csv("./dataset/WaterLevel.csv", low_memory=False)
        waterlevel = pd.read_csv("./dataset/scaled/WaterLevel_scaled.csv", low_memory=False)
        waterlevel.drop(['Unnamed: 0', 'date', 'time'], axis=1, inplace=True)

        # 아래 한 줄 강수량 피처 추가 코드
        self.data = pd.concat([rainfall, waterlevel], axis=1).values

        length = len(self.data)
        self.x_train = torch.tensor(self.data[:int(length*0.7)], dtype=torch.float32)
        self.y_train = self.x_train

        self.x_valid = torch.tensor(self.data[int(length*0.7):int(length*0.9)], dtype=torch.float32)
        self.y_valid = self.x_valid

        self.x_test = torch.tensor(self.data[int(length*0.9):], dtype=torch.float32)
        self.y_test = self.x_test

    def __len__(self):
        if self.mode == 'train':
            return len(self.x_train)

        if self.mode == 'valid':
            return len(self.x_valid)

        if self.mode == 'test':
            return len(self.x_test)

    def __getitem__(self, idx):
        x = None
        y = None

        if self.mode == 'train':
            x = self.x_train
            y = self.y_train

        if self.mode == 'valid':
            x = self.x_valid
            y = self.y_valid

        if self.mode == 'test':
            x = self.x_test
            y = self.y_test


        if (idx<len(x)) & (idx+self.window_size <len(x)):
            # tmp = torch.unsqueeze(y[idx:idx+self.window_size], 1)
            x = x[idx:idx + self.window_size]
            # x = torch.concat((x, tmp), axis=1)
            y = y[idx + self.window_size]
        elif (idx < len(x)) & (idx+self.window_size > len(x)):
            x = torch.rand(5, 1)
            y = 0

        return x, y

    def data_split(self):
        # train : valid : test = 7: 2: 1
        x_tr, x_tst, y_tr, y_tst = train_test_split(self.data, self.target, test_size=0.3, shuffle=False)
        x_vl, x_tst, y_vl, y_tst = train_test_split(x_tst, y_tst, test_size=0.3, shuffle=False)

        return x_tr, x_vl, x_tst, y_tr, y_vl, y_tst

