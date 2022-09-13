import torch
import torch.nn as nn
import pandas as pd


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#
#
#     def forward(self):
#
# class Decoder(nn.Module):
#     def ___init__(self):
#         super(Decoder, self).__init__()
#
# class ED_Model(nn.Module):
#     def __init__(self):
#         super(ED_Model, self).__init__()
#

class LSTM_Model(nn.Module):
    def __init__(self, window_size, input_size):
        super(LSTM_Model, self).__init__()
        self.window_size = window_size
        self.input_size = input_size            # the number of input features
        self.lstm = nn.LSTM(input_size=10, hidden_size=30, num_layers=1, batch_first=True)
        self.fc == nn.Linear(input_size, batch_first=True)

    def forward(self, X):
        h0 = torch.randn(self.window_size, 1, self.input_size)
        c0 = torch.randn(self.window_size, 1, self.input_size)
        y_pred, (h, c) = self.lstm(X, (h0, c0))
        y_pred = self.fc(y_pred)

