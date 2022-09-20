import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import LSTM_Model
from dataset import Data
# from torchinfo import summary
import os, sys

from datetime import datetime
from tqdm import trange, tqdm
from RMSE import RMSELoss
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


def train(tr_dataloader, vl_dataloader, epochs, model, loss_fn, optimizer, input_size):
    model.train()
    writer = SummaryWriter()

    for epoch in range(epochs):
        tr_loss_hist = 0.0
        i = 0

        print("[ EPOCH ", epoch, "]")

        for i, (X, y) in tqdm(enumerate(tr_dataloader)):
            if X.size(-1) != input_size:               # input size가 안맞는 부분이 있음.
                continue

            X = X.to(device)
            y = y.to(device)
            # print('\nㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡy.shape',y.shape)
            y_pred = model(X)
            # print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡy_pred.shape', y_pred.shape)
            loss = loss_fn(y, y_pred)
            # print(loss)
            tr_loss_hist += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_tr_loss = tr_loss_hist / float(i)
        writer.add_scalar("Mean train loss per epoch", mean_tr_loss, epoch)

        now = datetime.now()
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join('./checkpoint', now.strftime("%Y-%m-%d-%H-%M") + "-e" + str(epoch) + ".pt"))
            # torch.save(model.state_dict(), os.path.join("./saved_model", now.strftime("%Y-%m-%d-%H-%M") + "-e" + str(epoch) + ".pt"))

        print('start validation')
        mean_val_loss, output_lst, ans = validation(vl_dataloader, model, loss_fn)
        writer.add_scalar("Mean valid loss per epoch", mean_val_loss, epoch)

        print(len(output_lst))

    writer.flush()
    writer.close()


def validation(dataloader, model, loss_fn):
    model.eval()
    val_loss_hist = 0.0

    output_lst = []
    ans = []

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            # print(i)
            # print('hello')
            if X.size(-1) != input_size:               # input size가 안맞는 부분이 있음.
                continue

            ans.append(y)

            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            # print('val Im_model')
            # print('model calc')
            output_lst.append(y_pred.cpu())
            
            loss = loss_fn(y, y_pred)
            val_loss_hist += loss.item()

        val_loss = val_loss_hist / float(i)

    model.train()
    return val_loss, output_lst, ans


def test(dataloader, model, loss_fn):
    model.eval()
    ans = []
    test_loss_hist = 0.0
    output_lst = []

    with torch.no_grad():
        model.reset_hidden_state()
        for i, (X, y) in enumerate(dataloader):
            if X.size(-1) != input_size:               # input size가 안맞는 부분이 있음.
                continue
            ans.append(y)

            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            # print('model calc')
            output_lst.append(y_pred.cpu())
            loss = loss_fn(y, y_pred)
            test_loss_hist += loss.item()

        test_loss = test_loss_hist / float(i)


    return test_loss, output_lst, ans



if __name__ == '__main__':
    epoch = 100
    window_size = 5             # seq_len in nlp (L hyper-parameter)
    learning_rate = 1e-2
    weight_decay = 2e-5
    input_size = 6                     # feature 수, 즉 embedding size

    model = LSTM_Model(window_size, input_size).to(device)
    # print(summary(model, (1,5,11)))

    # loss_fn = torch.nn.MSELoss()
    loss_fn = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_dataset = Data(mode='train', window_size=window_size)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    valid_dataset = Data(mode='valid', window_size=window_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    test_dataset = Data(mode='test', window_size=window_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train(train_dataloader, valid_dataloader, epoch, model, loss_fn, optimizer, input_size)
    
    # test_loss, test_results, ans = test(test_dataloader, model, loss_fn)