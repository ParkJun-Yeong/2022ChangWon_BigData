import torch.nn as nn
import torch
import torch.nn.functional as F

class LRD(nn.Module): # Linear ReLU Dropout
    def __init__(self, in_ftrs, out_ftrs):
        super(LRD, self).__init__()

        self.Linear = nn.Linear(in_ftrs, out_ftrs)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p = 0.3)

class TDNN(nn.Module): # Time DNN
    def __init__(self):
        super(TDNN, self).__init__()

        self.Linear1 = LRD(512, 256)
        self.Linear2 = LRD(256, 128)
        self.Linear3 = LRD(128, 1)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
                
        return x