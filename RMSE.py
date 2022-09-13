import torch
import torch.nn as nn

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        x = x.squeeze()
        criterion = nn.MSELoss()
        
        eps = 1e-6
        loss = torch.sqrt(criterion(x,y) + eps)

        return loss
