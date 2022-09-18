import torch
import torch.nn as nn
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        self.hidden_size = 30
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1, bias=True).to(device)

    def forward(self, X):
        h0 = torch.randn(self.num_layers, 1, self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers, 1, self.hidden_size).to(device)          # input_size (1, window_size, input_size(feature))
        output, (h, c) = self.lstm(X, (h0, c0))                         # h size (1, 1, hidden size)
        h = torch.squeeze(h)
        y_pred = self.fc(h)
        y_pred = torch.mean(y_pred)
        y_pred = y_pred.unsqueeze(0)

        return y_pred

