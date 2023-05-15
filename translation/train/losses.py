import torch
import torch.nn as nn


class GroupsLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, probs):
        return -torch.log(self.softmax(probs)).mean(dim=-1)
