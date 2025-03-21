# models/model.py

import torch
import torch.nn as nn
import logging

class DuelingPokerNN(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.3)
        self.value_fc = nn.Linear(512, 256)
        self.value_ln = nn.LayerNorm(256)
        self.value_out = nn.Linear(256, 1)
        self.adv_fc = nn.Linear(512, 256)
        self.adv_ln = nn.LayerNorm(256)
        self.adv_out = nn.Linear(256, num_actions)

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Input size mismatch. Expected {self.input_size}, got {x.shape[-1]}")
        identity = x
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)) + identity[:, :512])
        value = self.value_out(torch.relu(self.value_ln(self.value_fc(x))))
        adv = self.adv_out(torch.relu(self.adv_ln(self.adv_fc(x))))
        return value + (adv - adv.mean(dim=1, keepdim=True))
