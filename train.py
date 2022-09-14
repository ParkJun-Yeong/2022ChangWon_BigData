import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import LSTM_Model
from dataset import Data
import os

from datetime import datetime
from tqdm import trange, tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


def train(tr_dataloader, vl_dataloader, epochs, model, loss_fn, optimizer):
    model.train()
    writer = SummaryWriter()

    for epoch in range(epochs):
        tr_loss_hist = 0.0
        i = 0

        print("[ EPOCH ", epoch, " ]")

        for i, (X, y) in tqdm(enumerate(tr_dataloader)):
            y_pred = model(X)
            loss = loss_fn(y, y_pred)
            tr_loss_hist += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_tr_loss = tr_loss_hist / float(i)
        writer.add_scalar("Mean train loss per epoch", mean_tr_loss, epoch)

        now = datetime.now()
        torch.save(model.state_dict(), os.path.join("./saved_model" + now.strftime("%Y-%m-%d-%H-%M") + "-e" + str(epoch) + ".pt"))

        mean_val_loss = validation(vl_dataloader, model, loss_fn)
        writer.add_scalar("Mean valid loss per epoch", mean_val_loss, epoch)

    writer.flush()
    writer.close()


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
    window_size = 5             # seq_len in nlp (L hyper-parameter)
    learning_rate = 1e-1
    weight_decay = 2e-5
    input_size = 10                     # feature 수, 즉 embedding size

    model = LSTM_Model(window_size, input_size)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset = Data(mode='train', window_size=window_size)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    valid_dataset = Data(mode='valid', window_size=window_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    test_dataset = Data(mode='test', window_size=window_size)
    test_dataloader = DataLoader(test_dataset, batch_size=window_size, shuffle=False)

    train(train_dataloader, valid_dataloader, epoch, model, loss_fn, optimizer)
