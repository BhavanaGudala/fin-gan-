import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.2):
        super().__init__()
        # Encoder
        self.enc_gru = nn.GRU(in_dim, hidden, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        
        # Decoder 
        self.dec_gru = nn.GRU(hidden, hidden, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        # Encode
        _, hidden_state = self.enc_gru(x)
        
        # Take the top layer's hidden state, repeat it for seq_len to decode
        context = hidden_state[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        h, _ = self.dec_gru(context)
        h = self.norm(h)
        h = self.dropout(h)
        return self.fc(h)

    def encode(self, x):
        """Extract encoder's top-layer hidden state (latent vector)."""
        _, hidden_state = self.enc_gru(x)
        return hidden_state[-1]  # (batch, hidden_dim)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over GRU hidden states.
    Instead of just taking the last hidden state (which loses information),
    this learns to weight all timesteps and produce a context vector.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        # h: (batch, seq_len, hidden)
        scores = self.attn(h)                          # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)         # (batch, seq_len, 1)
        context = (weights * h).sum(dim=1)             # (batch, hidden)
        return context


class Discriminator(nn.Module):
    """
    Discriminator (WGAN critic): scores how 'real' a network flow sequence is.
    Uses a 2-layer bidirectional GRU with attention pooling, LayerNorm,
    Dropout, and spectral normalization on the output layer.
    No sigmoid (Wasserstein objective).
    """
    def __init__(self, in_dim, hidden, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # bidirectional doubles the output dim
        self.norm = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionPooling(hidden * 2)
        self.fc = spectral_norm(nn.Linear(hidden * 2, 1))

    def forward(self, x):
        h, _ = self.gru(x)              # (batch, seq_len, hidden*2)
        h = self.norm(h)                # normalize
        h = self.dropout(h)             # regularize
        context = self.attention(h)     # (batch, hidden*2) — attention pooling
        return self.fc(context)         # (batch, 1) — no sigmoid
