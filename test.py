import torch
from torch.utils.data import DataLoader

from dataset import Data
from model import LSTM_Model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


def test(model, dataloader, loss_fn):
    model.eval()
    test_loss_hist = 0.0

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            if X.size(-1) != input_size:  # input size가 안맞는 부분이 있음.
                continue

            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            loss = torch.sqrt(loss_fn(y, y_pred))
            test_loss_hist += loss.item()

        test_loss = test_loss_hist / float(i)

    return test_loss

if __name__ == "__main__":
    window_size = 5             # seq_len in nlp (L hyper-parameter)
    input_size = 6
    loss_fn = torch.nn.MSELoss
    batch_size = 1

    test_dataset = Data(mode='test', window_size=window_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTM_Model(window_size, input_size, 30, batch_size)
    model.load_state_dict(torch.load("./checkpoint/2022-09-19-16-20-e450.pt")['model_state_dict'])

    test_loss = test(model, test_dataloader, loss_fn)

    print("Test Loss: ", test_loss)