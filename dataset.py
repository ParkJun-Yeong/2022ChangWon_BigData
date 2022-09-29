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

        # rainfall = pd.read_csv("./dataset/RainFall.csv", low_memory=False)

        # 아래 2줄 rainfall 을 피처로 추가하는 코드
        rainfall = pd.read_csv("./dataset/scaled/RainFall_scaled.csv", low_memory=False)
        rainfall.drop(['Unnamed: 단위(mm)', 'Unnamed: 0', 'date', 'time'], axis=1, inplace=True)


        # waterlevel = pd.read_csv("./dataset/WaterLevel.csv", low_memory=False)
        waterlevel = pd.read_csv("./dataset/scaled/WaterLevel_scaled.csv", low_memory=False)
        waterlevel.drop(['Unnamed: 0', 'date', 'time'], axis=1, inplace=True)

        # 아래 한 줄 강수량 피처 추가 코드
        # self.data = pd.concat([rainfall, waterlevel], axis=1)
        self.data = waterlevel


        # 창녕함안보 유입량이 더 패턴 있어 보이므로 타겟을 변경하기로 함.
        # target = pd.read_csv("./dataset/Target.csv", low_memory=False)
        # self.target = target['waterlevel']

        # target = pd.read_csv("./dataset/Target_discharge.csv", low_memory=False)
        # self.target = target['총유입량']

        inflow = pd.read_csv("./dataset/220928/Target_inflow_scaled.csv", low_memory=False).values.reshape(-1)
        self.inflow = inflow


        # self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = self.data_split()
        #
        # self.x_train = torch.tensor(self.x_train.values, dtype=torch.float32)
        # self.x_valid = torch.tensor(self.x_valid.values, dtype=torch.float32)
        # self.x_test = torch.tensor(self.x_test.values, dtype=torch.float32)
        #
        # self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32)
        # self.y_valid = torch.tensor(self.y_valid.values, dtype=torch.float32)
        # self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)


        length = len(self.inflow)
        self.x_train = torch.tensor(self.inflow[:int(length*0.7)], dtype=torch.float32)
        self.y_train = self.x_train

        self.x_valid = torch.tensor(self.inflow[int(length*0.7):int(length*0.9)], dtype=torch.float32)
        self.y_valid = self.x_valid

        self.x_test = torch.tensor(self.inflow[int(length*0.9):], dtype=torch.float32)
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

        if (idx< len(x)) & (idx+self.window_size*self.input_size < len(x)):
            # x = x[idx:idx + self.window_size]
            # y = y[idx + self.window_size]

            tmp_x = []
            tmp_y = []
            for i in range(self.window_size):
                tmp_x.append(x[idx+i:idx+i+self.input_size].unsqueeze(0))
                tmp_y.append(y[idx+i+self.input_size].unsqueeze(0))

            x = torch.concat(tmp_x)
            y = torch.concat(tmp_y)
        else:
            x = torch.rand(5,1)
            y = 0

        # if (idx<len(x)) & (idx+self.window_size <len(x)):
            # tmp = torch.unsqueeze(y[idx:idx+self.window_size], 1)
            # x = x[idx:idx + self.window_size]
            # x = torch.concat((x, tmp), axis=1)
            # y = y[idx + self.window_size]
        # elif (idx < len(x)) & (idx+self.window_size > len(x)):
        #
            # tmp = torch.unsqueeze(y[idx:len(x)], 1)
            # x = x[idx:len(x)]
            # x = torch.concat((x, tmp), axis=1)
            # if x.size(-1) == 0:
            #     return x, y[len(x)]
            # gap = self.window_size - len(x)
            # padding = torch.zeros(gap, x.size(-1))
            # x = torch.concat((x,padding))           # (5,11)로 맞춰주기
            #
            # padding = torch.zeros(gap)
            # y = y[len(x)]
            # y = y[len(x)].unsqueeze(-1)
            # y = torch.concat((y, padding))          # 5로 맞추기

        return x, y

    def data_split(self):
        # train : valid : test = 7: 2: 1
        x_tr, x_tst, y_tr, y_tst = train_test_split(self.data, self.target, test_size=0.3, shuffle=False)
        x_vl, x_tst, y_vl, y_tst = train_test_split(x_tst, y_tst, test_size=0.3, shuffle=False)

        return x_tr, x_vl, x_tst, y_tr, y_vl, y_tst

