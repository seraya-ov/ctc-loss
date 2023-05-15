import torch
import torch.nn as nn

from attention import Attention, GatedAttention


class LengthClassDuplicationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_duplication=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(hidden_dim, max_duplication)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.max_duplication = max_duplication

    def forward(self, x):
        lengths_probs = self.projection(
            self.dropout(self.ln(self.relu(self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)))))
        lengths = torch.cumsum(torch.argmax(lengths_probs, dim=-1).unsqueeze(1) + 1, dim=-1)
        duplication_matrix = torch.zeros((x.shape[0], x.shape[1] * self.max_duplication, x.shape[1])) + torch.arange(1,
                                                                                                                     1 +
                                                                                                                     x.shape[
                                                                                                                         1] * self.max_duplication).unsqueeze(
            0).unsqueeze(2)
        duplication_matrix = duplication_matrix.to(device=x.device)
        duplication_matrix[:, :, 0] = torch.clamp(lengths[:, :, 0] - duplication_matrix[:, :, 0] + 1, 0, 1)
        duplication_matrix[:, :, 1:] = torch.clamp(lengths[:, :, 1:] - duplication_matrix[:, :, 1:] + 1, 0,
                                                   1) * torch.clamp(duplication_matrix[:, :, 1:] - lengths[:, :, :-1],
                                                                    0, 1)

        return lengths_probs, duplication_matrix


class LengthRegDuplicationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_duplication=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(hidden_dim, 1)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.max_duplication = max_duplication

    def forward(self, x):
        lengths = torch.clamp(
            self.projection(self.dropout(self.ln(self.relu(self.conv(x.permute(0, 2, 1)).permute(0, 2, 1))))).squeeze(
                -1), 0, self.max_duplication - 1)
        lengths = torch.cumsum(lengths.floor().unsqueeze(1) + 1, dim=-1)
        duplication_matrix = torch.zeros((x.shape[0], x.shape[1] * self.max_duplication, x.shape[1])) + torch.arange(1,
                                                                                                                     1 +
                                                                                                                     x.shape[
                                                                                                                         1] * self.max_duplication).unsqueeze(
            0).unsqueeze(2)
        duplication_matrix = duplication_matrix.to(device=x.device)
        duplication_matrix[:, :, 0] = torch.clamp(lengths[:, :, 0] - duplication_matrix[:, :, 0] + 1, 0, 1)
        duplication_matrix[:, :, 1:] = torch.clamp(lengths[:, :, 1:] - duplication_matrix[:, :, 1:] + 1, 0,
                                                   1) * torch.clamp(duplication_matrix[:, :, 1:] - lengths[:, :, :-1],
                                                                    0, 1)
        return duplication_matrix


class GroupingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(hidden_dim, 2)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        groups_probs = self.projection(self.dropout(self.ln(self.relu(self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)))))
        groups = torch.argmax(groups_probs, dim=-1)
        groups[:, 0] = 0
        start_groups = groups.nonzero(as_tuple=True)
        last_group = groups.shape[1]
        groups = torch.cumsum(torch.ones(groups.shape).to(device=x.device), dim=-1)
        groups[:, 0] = 0
        groups[start_groups[0], start_groups[1]] = 0
        groups[:, -1] = last_group
        groups = torch.gather(groups, 1,
                              groups.ne(0).to(dtype=torch.int).argsort(dim=1, descending=True, stable=True)).unsqueeze(
            2)
        grouping_matrix = torch.zeros((x.shape[0], x.shape[1], x.shape[1])) + torch.arange(1, 1 + x.shape[1]).unsqueeze(
            0).unsqueeze(1)
        grouping_matrix = grouping_matrix.to(device=x.device)
        grouping_matrix[:, 0, :] = torch.clamp(groups[:, 0, :] - grouping_matrix[:, 0, :] + 1, 0, 1)
        grouping_matrix[:, 1:, :] = torch.clamp(groups[:, 1:, :] - grouping_matrix[:, 1:, :] + 1, 0, 1) * torch.clamp(
            grouping_matrix[:, 1:, :] - groups[:, :-1, :], 0, 1)

        return groups_probs, grouping_matrix


class PermutationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim), Attention(hidden_dim, attention_dim),
            nn.Linear(hidden_dim, hidden_dim), Attention(hidden_dim, attention_dim)])
        self.pre_net = nn.Linear(hidden_dim, hidden_dim)
        self.gated_attention = GatedAttention(hidden_dim, attention_dim)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return self.gated_attention(self.pre_net(x))
