import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import LSTM_Model
from dataset import Data
# from torchinfo import summary
import os

from datetime import datetime
from tqdm import trange, tqdm

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("device: ", device)


def train(tr_dataloader, vl_dataloader, epochs, model, loss_fn, optimizer, input_size):
    model.train()
    writer = SummaryWriter()

    for epoch in range(epochs):
        tr_loss_hist = []
        i = 0

        print("[ EPOCH ", epoch, " ]")

        for i, (X, y) in tqdm(enumerate(tr_dataloader)):
            if X.size(-1) != input_size:               # input size가 안맞는 부분이 있음.
                continue

            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            loss = torch.sqrt(loss_fn(y, y_pred))
            # print("y: ", y.shape)
            # print("y_pred: ", y_pred.shape)
            # loss = loss_fn(y, y_pred)

            # tr_loss_hist += loss.item()
            tr_loss_hist.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tr_loss_hist = torch.tensor(tr_loss_hist)
        mean_tr_loss = torch.mean(tr_loss_hist)
        writer.add_scalar("Mean train RMSE loss per epoch", mean_tr_loss, epoch)
        print("Train Loss: ", mean_tr_loss)
        print("Train Loss History: ", tr_loss_hist)

        now = datetime.now()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join('./checkpoint', now.strftime("%Y-%m-%d-%H-%M") + "-e" + str(epoch) + ".pt"))
        # torch.save(model.state_dict(), os.path.join("./saved_model", now.strftime("%Y-%m-%d-%H-%M") + "-e" + str(epoch) + ".pt"))

        mean_val_loss = validation(vl_dataloader, model, loss_fn)
        writer.add_scalar("Mean valid RMSE loss per epoch", mean_val_loss, epoch)
        print("Valid Loss: ", mean_val_loss)

    writer.flush()
    writer.close()


def validation(dataloader, model, loss_fn):
    model.eval()
    val_loss_hist = []

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            if X.size(-1) != input_size:               # input size가 안맞는 부분이 있음.
                continue

            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            loss = torch.sqrt(loss_fn(y, y_pred))
            # loss = loss_fn(y, y_pred)
            # val_loss_hist += loss.item()
            val_loss_hist.append(loss.item())

        val_loss_hist = torch.tensor(val_loss_hist)
        val_loss = torch.mean(val_loss_hist)
        print("Valid Loss History: ", val_loss_hist)

    model.train()

    return val_loss


if __name__ == '__main__':
    epoch = 400
    window_size = 5             # seq_len in nlp (L hyper-parameter)
    learning_rate = 1e-4
    weight_decay = 2e-5
    input_size = 5                     # feature 수, 즉 embedding size
    batch_size = 1

    model = LSTM_Model(window_size=window_size, input_size=input_size, hidden_size=10, batch_size=batch_size).to(device)
    # print(summary(model, (window_size,input_size), batch_dim=batch_size))

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset = Data(mode='train', window_size=window_size, input_size=input_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    valid_dataset = Data(mode='valid', window_size=window_size, input_size=input_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Data(mode='test', window_size=window_size, input_size=input_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train(train_dataloader, valid_dataloader, epoch, model, loss_fn, optimizer, input_size)
