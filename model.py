import torch
import torch.nn as nn
import pandas as pd

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


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
#     def __init__(self, window_size, input_size, hidden_size, batch_size):
#         super(ED_Model, self).__init__()
#         self.encoder = LSTM_Model(window_size, input_size, hidden_size, batch_size)
#         self.decoder = LSTM_Model(1, 1, )


class LSTM_Model(nn.Module):
    def __init__(self, window_size, input_size, hidden_size, batch_size):
        super(LSTM_Model, self).__init__()
        self.window_size = window_size
        self.input_size = input_size            # the number of input features
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1, bias=True).to(device)

    def forward(self, X):
        h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)          # input_size (1, window_size, input_size(feature))
        output, (h, c) = self.lstm(X, (h0, c0))
        output = torch.mean(output, axis=1).unsqueeze(0)# h size (1, 1, hidden size)
        # h = torch.squeeze(h)
        y_pred = self.fc(output)

        if self.num_layers > 1:
            y_pred = torch.mean(y_pred)

        # if len(y_pred.size()) < 1:
        #     y_pred = y_pred.unsqueeze(0)

        while len(y_pred.size()) > 1:
            y_pred = y_pred.squeeze(-1)

        return y_pred

