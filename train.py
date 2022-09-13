import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import LSTM_Model
from dataset import Data

from datetime import datetime
from tqdm import trange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


def train(dataloader, epochs, model, loss_fn, optimizer):
    model.train()
    tr_loss_hist = []
    vl_loss_hist = []
    writer = SummaryWriter()

    for epoch in trange(epochs):
        print("[ EPOCH ", epoch, " ]")

        for i, (X, y) in enumerate(dataloader):
            y_pred = model(X)
            loss = loss_fn(y, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




if __name__ == '__main__':
    epoch = 100
    window_size = 7             # batch_size = num_layers of lstm
    learning_rate = 1e-1
    weight_decay = 2e-5

    model = LSTM_Model()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset = Data(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=window_size, shuffle=False)

    valid_dataset = Data(mode='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=window_size, shuffle=False)

    test_dataset = Data(mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=window_size, shuffle=False)

    train(train_dataloader, epoch, model, loss_fn, optimizer)