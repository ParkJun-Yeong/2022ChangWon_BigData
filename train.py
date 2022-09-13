import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import LSTM_Model
from dataset import Data
import os

from datetime import datetime
from tqdm import trange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


def train(tr_dataloader, vl_dataloader, epochs, model, loss_fn, optimizer):
    model.train()
    writer = SummaryWriter()

    for epoch in trange(epochs):
        tr_loss_hist = 0.0
        i = 0

        print("[ EPOCH ", epoch, " ]")

        for i, (X, y) in enumerate(tr_dataloader):
            y_pred = model(X)
            loss = loss_fn(y, y_pred)
            tr_loss_hist += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_tr_loss = tr_loss_hist / float(i)
        writer.add_scalar("Mean train loss per epoch", mean_tr_loss, epoch)

        now = datetime.now()
        torch.save(model.state_dict(), os.path.join("./saved_model" + now.strftime("%Y-%m-%d-%H-%M") + "-e" + epoch + ".pt"))

        mean_val_loss = validation(vl_dataloader, model, loss_fn)
        writer.add_scalar("Mean valid loss per epoch", mean_val_loss, epoch)


def validation(dataloader, model, loss_fn):
    model.eval()
    val_loss_hist = 0.0

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            y_pred = model(X)
            loss = loss_fn(y, y_pred)
            val_loss_hist += loss.item()

        val_loss = val_loss_hist / float(i)

    model.train()

    return val_loss


if __name__ == '__main__':
    epoch = 100
    window_size = 7             # batch_size, so a size of single input is (window_size, 1, 10). it is the same concept of 'seq_len' in nlp
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

    train(train_dataloader, valid_dataloader, epoch, model, loss_fn, optimizer)