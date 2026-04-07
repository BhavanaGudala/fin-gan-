# Architecture Improvements & Rationale

This document explains the changes made to the GAN-based anomaly detection system to improve ROC-AUC performance from ~90% towards the ~97% reported in the reference paper. All changes are standard deep learning best practices and do **not** replicate the paper's TCN/Self-Attention approach — our system remains a **GRU-based WGAN-GP**, which is a fundamentally different architecture.

---

## 1. Model Architecture (`src/model.py`)

### 1.1 Multi-Layer GRU (1 layer → 2 layers)

**Before:**
```python
self.gru = nn.GRU(noise_dim, hidden, batch_first=True)
```

**After:**
```python
self.gru = nn.GRU(
    noise_dim, hidden,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)
```

**Why:** A single GRU layer has limited capacity to learn the complex temporal patterns present in network flow data. Stacking two layers allows the network to learn hierarchical representations — the first layer captures low-level features (e.g., packet timing patterns), while the second layer captures higher-level abstractions (e.g., flow behavior over the full window). This is standard practice in RNN-based architectures and is analogous to using deeper convolutional networks.

---

### 1.2 Layer Normalization

**Before:** No normalization.

**After (in both Generator and Discriminator):**
```python
self.norm = nn.LayerNorm(hidden)
# ...
h = self.norm(h)
```

**Why:** Without normalization, the internal activations of the GRU can drift to extreme values during training, causing unstable gradients. Layer normalization standardizes activations at each timestep, which:
- Stabilizes training dynamics
- Allows the use of higher learning rates
- Reduces sensitivity to weight initialization
- Helps the model converge to better solutions

This is especially important in WGAN training, where the critic and generator are constantly competing — instability in one destabilizes the other.

---

### 1.3 Dropout Regularization

**Before:** No dropout anywhere.

**After (in both Generator and Discriminator):**
```python
self.dropout = nn.Dropout(0.2)
# ...
h = self.dropout(h)
```

**Why:** The original model had no regularization, making it prone to **overfitting** — the discriminator memorizes specific patterns in the training data rather than learning generalizable features of "normal" traffic. Dropout randomly zeroes 20% of activations during training, forcing the network to use redundant representations and preventing any single neuron from becoming overly relied upon. This directly improves the discriminator's ability to generalize to unseen test data, which is critical for anomaly detection.

---

### 1.4 Bidirectional GRU in Discriminator

**Before:**
```python
self.gru = nn.GRU(in_dim, hidden, batch_first=True)
```

**After:**
```python
self.gru = nn.GRU(
    in_dim, hidden,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.2
)
```

**Why:** A unidirectional GRU processes the sequence left-to-right only, meaning the hidden state at timestep `t` only knows about flows `0..t`. However, in anomaly detection, a flow at timestep `t=3` might only be identifiable as anomalous when considering flows that come *after* it (e.g., a sudden drop in traffic at `t=7`). A bidirectional GRU processes the sequence in both directions and concatenates the hidden states, giving the discriminator a complete view of the temporal context at every timestep.

This is a key differentiator from the reference paper, which uses neither bidirectional processing nor GRU-based architectures.

---

### 1.5 Attention Pooling (replaces last-hidden-state)

**Before:**
```python
def forward(self, x):
    h, _ = self.gru(x)
    last_hidden = h[:, -1]         # only uses the LAST timestep
    return self.fc(last_hidden)
```

**After:**
```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        scores = self.attn(h)                    # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)   # learned importance per timestep
        context = (weights * h).sum(dim=1)       # weighted sum
        return context
```

```python
# In Discriminator.forward():
context = self.attention(h)     # replaces h[:, -1]
return self.fc(context)
```

**Why:** Taking only the last hidden state (`h[:, -1]`) discards information from earlier timesteps. In a 10-step window of network flows, the anomalous pattern might appear at any position — not just the end. Attention pooling learns a set of importance weights over all timesteps, allowing the discriminator to focus on the most informative parts of the sequence. For example, if a burst of abnormal packets occurs at timestep 4, the attention mechanism can assign a high weight to that position.

This is another key differentiator — the reference paper does not use attention pooling over GRU outputs.

---

## 2. Training Loop (`src/train.py`)

### 2.1 Critic Training Ratio (1:1 → 5:1)

**Before:**
```python
for x, _ in loader:
    # Train D once
    # Train G once
```

**After:**
```python
for i, (x, _) in enumerate(loader):
    # Train D every step
    # ...

    # Train G only every 5th step
    if (i + 1) % n_critic == 0:
        # ...
```

**Why:** In WGAN-GP, the critic needs to be a good approximation of the Wasserstein distance for the generator to receive meaningful gradients. If the critic is undertrained (1:1 ratio), it provides noisy, unreliable feedback to the generator, leading to poor convergence. Training the critic 5 times for every generator update (the ratio recommended in the original WGAN paper by Arjovsky et al., 2017) ensures the critic stays ahead and provides stable, informative gradients.

---

### 2.2 Early Stopping with Patience

**Before:** Fixed 20 epochs, no early stopping.

**After:**
```python
patience_counter = 0

for epoch in range(cfg["epochs"]):     # up to 50 epochs
    # ...
    if avg_d_loss < best_loss:
        best_loss = avg_d_loss
        torch.save(D.state_dict(), "checkpoints/best_D.pth")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:   # patience = 10
        print(f"Early stopping at epoch {epoch+1}")
        break
```

**Why:** Without early stopping, the model trains for a fixed number of epochs regardless of whether it has converged or begun overfitting. With early stopping:
- Training automatically stops when the model stops improving
- The best checkpoint (not the last) is saved
- Overfitting is avoided — the discriminator doesn't memorize training data
- Training time is not wasted on unproductive epochs

---

### 2.3 Checkpoint Name Fix

**Before:** Training saves `best_D.pth` and `D_last.pth`, but inference loads `D.pth` — causing a `FileNotFoundError`.

**After:** Training saves `best_D.pth`, `D_last.pth`, and also `D.pth`. Inference loads `best_D.pth` directly.

**Why:** This was a bug. The best-performing model was never being used for inference.

---

## 3. Configuration (`configs/config.yaml`)

### 3.1 Increased Hidden Dimension (64 → 128)

**Before:** `hidden_dim: 64` (implicit)

**After:** `hidden_dim: 128`

**Why:** With 76 input features, a hidden dimension of 64 compresses the representation too aggressively. Increasing to 128 gives the GRU more capacity to represent the feature space without losing information. Combined with the bidirectional GRU (which doubles the effective hidden size to 256), the discriminator now has sufficient capacity to learn the complex boundary between normal and anomalous traffic.

---

### 3.2 Increased Epochs (20 → 50) with Early Stopping

**Why:** 20 epochs may not be enough for convergence, especially with the 5:1 critic ratio (the generator effectively sees only 1/5th of the updates). Increasing to 50 gives more room to converge, while early stopping (patience=10) ensures we don't overtrain.

---

## 4. Inference (`src/infer.py`)

### 4.1 Config-Driven Model Instantiation

**Before:**
```python
D = Discriminator(loader.dataset.data.shape[1], 64)
```

**After:**
```python
hidden_dim = cfg.get("hidden_dim", 64)
num_layers = cfg.get("num_layers", 2)
dropout = cfg.get("dropout", 0.2)
D = Discriminator(loader.dataset.data.shape[1], hidden_dim, num_layers, dropout)
```

**Why:** The model architecture must match exactly between training and inference. Hardcoding `64` would cause a shape mismatch error when loading weights from a model trained with `hidden_dim=128`.

### 4.2 Device-Safe Checkpoint Loading

**Before:**
```python
D.load_state_dict(torch.load("checkpoints/D.pth"))
```

**After:**
```python
D.load_state_dict(torch.load("checkpoints/best_D.pth", map_location=device))
```

**Why:** If training runs on GPU but inference runs on CPU (or vice versa), `torch.load` fails without `map_location`. This ensures the checkpoint loads correctly regardless of the device.

---

## Summary of Changes

| File | Change | Category |
|------|--------|----------|
| `src/model.py` | 2-layer GRU | Model capacity |
| `src/model.py` | LayerNorm | Training stability |
| `src/model.py` | Dropout (0.2) | Regularization |
| `src/model.py` | Bidirectional GRU (Discriminator) | Better temporal modeling |
| `src/model.py` | Attention pooling (Discriminator) | Better sequence aggregation |
| `src/train.py` | 5:1 critic training ratio | WGAN best practice |
| `src/train.py` | Early stopping (patience=10) | Prevent overfitting |
| `src/train.py` | Checkpoint name fix | Bug fix |
| `configs/config.yaml` | hidden_dim: 128 | Model capacity |
| `configs/config.yaml` | epochs: 50 | Training budget |
| `configs/config.yaml` | Added n_critic, dropout, patience | New parameters |
| `src/infer.py` | Config-driven model params | Consistency |
| `src/infer.py` | map_location in torch.load | Device compatibility |

All changes are standard deep learning best practices. The architecture remains a **GRU-based WGAN-GP** — fundamentally different from the reference paper's TCN/Self-Attention approach.
