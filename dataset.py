from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os, sys

class Data(Dataset):
    def __init__(self, mode, window_size):
        super(Data, self).__init__()

        self.mode = mode
        self.window_size = window_size
        # rainfall = pd.read_csv("./dataset/RainFall.csv", low_memory=False)
        rainfall = pd.read_csv("./dataset/Rainy_season/rs_RainFall.csv", low_memory=False)
        # rainfall.drop(['Unnamed: 단위(mm)', 'Unnamed: 0', 'date', 'time'], axis=1, inplace=True)
        rainfall = rainfall['길곡강수량']
        # waterlevel = pd.read_csv("./dataset/WaterLevel.csv", low_memory=False)
        waterlevel = pd.read_csv("./dataset/Rainy_season/rs_WaterLevel.csv", low_memory=False)
        waterlevel.drop(['Unnamed: 0', 'date', 'time'], axis=1, inplace=True)
        self.data = pd.concat([rainfall, waterlevel], axis=1)

        # 창녕함안보 유입량이 더 패턴 있어 보이므로 타겟을 변경하기로 함.
        # target = pd.read_csv("./dataset/Target.csv", low_memory=False)
        # self.target = target['waterlevel']
        target = pd.read_csv("./dataset/Rainy_season/Target.csv", low_memory=False)
        self.target = target['총유입량']

        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = self.data_split()

        self.x_train = torch.tensor(self.x_train.values, dtype=torch.float32)
        self.x_valid = torch.tensor(self.x_valid.values, dtype=torch.float32)
        self.x_test = torch.tensor(self.x_test.values, dtype=torch.float32)

        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32)
        self.y_valid = torch.tensor(self.y_valid.values, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)


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

        if (idx < len(x)) & (idx+self.window_size <len(x)):
            tmp = torch.unsqueeze(y[idx:idx+self.window_size], 1)
            # print('temtem', tmp.shape)
            # print('X before concat :', x.shape)
            # x = torch.concat((x[idx:idx + self.window_size], tmp), axis=1)
            x = x[idx : idx + self.window_size]
            # print('X in if :', x.shape)
            y = y[idx + self.window_size]
        elif (idx < len(x)) & (idx+self.window_size > len(x)):
            tmp = torch.unsqueeze(y[idx:len(x)], 1)
            x = torch.concat((x[idx:len(x)], tmp), axis=1)
            # print('X in elif :', x.shape)
            y = y[len(x) + self.window_size]

        # print('x : ',x.shape)
        # print('y : ',y.shape)
        return x, y

    def data_split(self):
        # train : valid : test = 7: 2: 1
        x_tr, x_tst, y_tr, y_tst = train_test_split(self.data, self.target, test_size=0.3, shuffle=False)
        x_vl, x_tst, y_vl, y_tst = train_test_split(x_tst, y_tst, test_size=0.3, shuffle=False)

        return x_tr, x_vl, x_tst, y_tr, y_vl, y_tst

