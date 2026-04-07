import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden, out_dim):
        super().__init__()
        self.gru = nn.GRU(noise_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, z):
        h, _ = self.gru(z)
        return self.fc(h)

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        last_hidden = h[:, -1]
        return self.fc(last_hidden)   # No sigmoid
