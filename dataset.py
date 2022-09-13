from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


class Data(Dataset):
    def __init__(self, mode):
        super(Data, self).__init__()

        self.mode = mode
        rainfall = pd.read_csv("./dataset/RainFall.csv")
        rainfall.drop(['Unnamed: 0', 'date', 'time'], axis=1, inplace=True)
        waterfall = pd.read_csv("./dataset/WaterLevel.csv")
        waterfall.drop(['Unnamed: 0', 'date', 'time'], axis=1, inplace=True)
        self.data = pd.concat([rainfall, waterfall], axis=1)

        target = pd.read_csv("./dataset/Target.csv")
        self.target = target['waterlevel']

        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = self.data_split()

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
            x = torch.tensor(self.x_train.iloc[idx].values)
            y = torch.tensor(self.y_train.iloc[idx].values)

        if self.mode == 'valid':
            x = torch.tensor(self.x_valid.iloc[idx].values)
            y = torch.tensor(self.y_valid.iloc[idx].values)

        if self.mode == 'test':
            x = torch.tensor(self.x_test.iloc[idx].values)
            y = torch.tensor(self.y_test.iloc[idx].values)

        return x, y

    def data_split(self):
        # train : valid : test = 7: 2: 1
        x_tr, x_tst, y_tr, y_tst = train_test_split(self.data, self.target, test_size=0.3, shuffle=False)
        x_vl, x_tst, y_vl, y_tst = train_test_split(x_tst, y_tst, test_size=0.3, shuffle=False)

        return x_tr, x_vl, x_tst, y_tr, y_vl, y_tst

