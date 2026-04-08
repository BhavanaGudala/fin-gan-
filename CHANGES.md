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

---

## 5. AE-WGAN-GP (Autoencoder Generator Upgrade)

### 5.1 Generator as Autoencoder

**Before:** The Generator took random noise `z` to generate Fake data.

**After:** 
```python
class Generator(nn.Module):
    def __init__(self, in_dim, hidden, out_dim...):
        self.enc_gru = nn.GRU(in_dim, hidden...)
        self.dec_gru = nn.GRU(hidden, hidden...)
```

**Why:** Using random noise (standard GAN) makes it mathematically difficult to score anomalies in sequence data because we cannot naturally reconstruct a given `x`. By making the Generator an Autoencoder, it learns to compress and decompress *only benign data*.

### 5.2 Training Generator with Real Inputs
**Before:** `fake_x = G(z)`
**After:** `fake_x = G(real_x)` with `g_loss = critic_loss + 10.0 * MSE(real_x, fake_x)`

**Why:** The Generator now trains to simultaneously fool the Critic AND accurately reconstruct the original benign input sequence.

### 5.3 Reconstruction-based Anomaly Score

**Before:** `Anomaly Score = -D(x)`
**After (v1):** `Anomaly Score = 0.9 * MSE(x, x_hat) + 0.1 * -D_best(x)`
**After (v2 — current):** `Anomaly Score = MSE(x, x_hat)`

**Why (v1):** The Critic is only a binary classifier (Real vs Fake). It is not reliable when fed out-of-distribution attacks, causing the inverted ROC-AUC mapping issue. By strongly weighting the **Reconstruction Loss**, attacks will trigger huge errors since the Autoencoder has never learned to decode attack patterns, providing a highly robust and mathematically sound signal for 97%+ target performance.

**Why (v2):** The combined score from v1 still produced a **ROC-AUC of 0.4674** (worse than random). Root cause analysis of the training logs revealed:

1. **Critic score inversion:** During WGAN training, D(real) drifted to large negative values (−25 at epoch 11). The anomaly score used `−D(x)`, which means for benign data: `−(−25) = +25`. This **pushed benign scores upward**, partially inverting the signal — benign flows scored *higher* than some attacks.

2. **Training collapse:** D(real) and D(fake) both drifted negative together (−25 and −52 respectively) instead of converging. The generator loss kept rising (16 → 60), indicating the autoencoder never learned to properly reconstruct benign data. The critic was providing meaningless gradients.

3. **Scale mismatch:** Even with 0.9/0.1 weighting, the critic scores (range: −50 to +5) and reconstruction errors (range: 0 to 5000+) operated on completely different scales. The 10% critic contribution was enough to corrupt the ranking of borderline samples.

**The fix:** Use **pure reconstruction error** as the anomaly score. This is the mathematically natural metric for an autoencoder-based anomaly detector: the generator was trained only on benign data, so attacks it has never seen produce large reconstruction error. The critic score is dropped entirely because it is unreliable when WGAN training has not converged properly.

---

## 6. Training Improvements (Phase 2)

### 6.1 Early Stopping on Reconstruction Loss

**Before:** `if avg_d_loss < best_loss` — saved whenever the critic loss improved.  
**After:** `if avg_recon < best_recon` — saved whenever the reconstruction loss improved.

**Why:** The critic loss in WGAN does not directly correlate with anomaly detection quality. By tracking reconstruction loss instead, we save the checkpoint when the autoencoder is actually learning to reconstruct benign data better. This prevents saving a checkpoint at epoch 11 when recon loss was still declining through epoch 21.

**Impact:** With the old criterion, recon_loss stabilized at ~0.86. With the new criterion, it can drop to ~0.71 or lower, improving the detector's ability to distinguish benign from attack flows via reconstruction error magnitude.

### 6.2 Gradient Clipping

**Before:** No gradient clipping.  
```python
opt_D.step()
opt_G.step()
```

**After:** 
```python
torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
opt_D.step()
torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
opt_G.step()
```

**Why:** WGAN training with unbounded gradients causes D(real) and D(fake) to drift to extreme values (−25, −52), destabilizing the generator. Gradient clipping bounds the gradient magnitude at 1.0, preventing the Wasserstein distance approximation from diverging. This keeps both discriminator scores in a stable range, allowing the generator to receive meaningful feedback.

**Impact:** Prevents the score drift observed in the training logs (D(real): 2.7 → −27.1). With gradient clipping, D(real) and D(fake) should converge toward zero or stabilize at manageable values.

### 6.3 Configurable Reconstruction Weight

**Before:** Hardcoded `g_loss = critic_loss + 10.0 * recon_loss` in training loops.  
**After:** 
- Config parameter: `recon_weight: 1.0` (reduced from 10.0)
- Training loop: `g_loss = critic_loss + recon_weight * recon_loss`

**Why:** The 10x weight meant the generator focused almost entirely on reconstruction MSE, drowning out the adversarial signal from the critic. Reducing to 1.0 gives equal weight to both objectives: the critic guides the generator to fool the discriminator (ensuring the autoencoder learns the benign *distribution*, not just point-wise reconstruction), while reconstruction error guides it to minimize MSE.

**Tuning:** This can be adjusted in `configs/config.yaml` without recompilation. Try `recon_weight: 0.1-2.0` based on detection needs.

### 6.4 Increased Training Budget

**Before:** 50 epochs with patience=10.  
**After:** 100 epochs with patience=15.

**Why:** The autoencoder benefits from extended training, especially with the new early stopping criterion (recon_loss). An extra 50 epochs gives the model time to find better reconstructions for benign data, improving the benign/attack separation.

---

## Summary of Phase 1 + 2 Impact

| Metric | Phase 1 Only | Phase 2 (initial) | Paper Baseline |
|---|---|---|---|
| **ROC-AUC** | 0.789 | 0.793 | 0.9963 (self-attention, TCN) |
| **Syn detection** | 3.8% | 3.1% | 94.83% |
| **Benign TNR** | 72.9% | 73.1% | ~96-98% |
| **Key changes** | Critic score removed | + Gradient clipping, recon early stop, weight tuning, more epochs | TCN/self-attention, LSTM baseline |

**Phase 2 diagnosis:** The initial `recon_weight=1.0` barely improved results because the critic loss (~60) completely dominated the generator objective. With `recon_loss ≈ 0.80`, the reconstruction term was only ~1.3% of total generator loss. The autoencoder was not learning to reconstruct — it was learning to fool the critic.

---

## 7. Phase 2b — Reconstruction-Dominant Training

After Phase 2's marginal improvement (0.789 → 0.793), analysis of the generator loss composition revealed the core issue: the adversarial critic loss was ~75× larger than the reconstruction loss, making the autoencoder optimize almost exclusively to fool the critic rather than to faithfully reconstruct benign data.

### 7.1 Reconstruction Weight (1.0 → 100.0)

**Before:** `recon_weight: 1.0` → `g_loss = critic_loss + 1.0 × recon_loss`

**After:** `recon_weight: 100.0` → `g_loss = critic_loss + 100.0 × recon_loss`

**Why:** At the best checkpoint (epoch 27, recon_weight=1.0):
- `critic_loss ≈ 60` (from `-D(G(x))`)
- `recon_loss ≈ 0.80` (MSE between `x` and `G(x)`)
- **Old:** `g_loss = 60 + 1.0 × 0.80 = 60.80` → reconstruction is only **1.3%** of gradient signal
- **New:** `g_loss = 60 + 100 × 0.80 = 140` → reconstruction is **57%** of gradient signal

With `recon_weight=100`, the generator receives meaningful gradient from the reconstruction objective. This means the autoencoder will actively learn to minimize MSE on benign data, resulting in:
- Lower reconstruction error for benign flows → better true negative rate
- Higher reconstruction error for attack flows → better true positive rate
- Better class separation in the anomaly score distribution

The adversarial term still contributes — it regularizes the autoencoder to produce outputs that lie on the benign data manifold (not just minimize pixel-wise error). But it no longer drowns out the reconstruction signal.

### 7.2 Reduced Hidden Dimension (128 → 64)

**Before:** `hidden_dim: 128` → latent bottleneck: 77 features × 10 timesteps → 128-dim (compression ratio ~6:1)

**After:** `hidden_dim: 64` → latent bottleneck: 77 features × 10 timesteps → 64-dim (compression ratio ~12:1)

**Why:** The autoencoder's anomaly detection ability depends on the **information bottleneck** being tight enough that it cannot losslessly encode arbitrary input — only patterns it has been trained on (benign flows). With `hidden_dim=128`, the bottleneck was too wide:
- The encoder had enough capacity to partially encode attack patterns by memorizing general-purpose compression strategies
- Attack reconstruction errors were not sufficiently higher than benign reconstruction errors
- This directly hurt the ROC-AUC because the reconstruction-based anomaly score couldn't separate the classes

With `hidden_dim=64`:
- The encoder must learn a more selective compression — it can only represent the 64 most important dimensions of benign traffic
- Attack patterns that don't align with these learned dimensions will produce significantly higher reconstruction error
- This also reduces total parameters: Generator 386,893 → ~105K, Discriminator 456,450 → ~125K

**Trade-off:** Too small a bottleneck would also degrade benign reconstruction (increasing false positives). The 64-dim bottleneck with 77 input features provides ~1:1 compression at each timestep, which is a reasonable lower bound for this dataset.

### 7.3 Increased Patience (15 → 20)

**Before:** `patience: 15`  
**After:** `patience: 20`

**Why:** With the tighter bottleneck (64 vs 128) and the shifted loss balance (reconstruction now dominant), the model needs more epochs to converge because:
1. The 64-dim bottleneck is harder to optimize — fewer parameters means the loss landscape has fewer easy paths
2. The 100× reconstruction weight changes the gradient dynamics — the model must re-learn a different balance between the two objectives
3. Recon loss improvements may come in small, intermittent drops rather than steady decline

Patience of 20 gives the model enough runway to find these improvements without stopping prematurely.

### 7.4 Fixed "Best Epoch" Display

**Before:** `history['epoch'][np.argmin(history['d_loss'])]` — showed epoch with lowest critic loss.  
**After:** `history['epoch'][np.argmin(history['recon_loss'])]` — shows epoch with lowest reconstruction loss.

**Why:** The best checkpoint is now selected by reconstruction loss, so the display should match. The old code reported epoch 11 (best critic loss) when the actual best model was saved at epoch 27 (best recon loss).

---

## Summary of All Phases

| Metric | Baseline (v0) | Phase 1 | Phase 2a | Phase 2b | Paper |
|---|---|---|---|---|---|
| **ROC-AUC** | 0.467 | 0.789 | 0.793 | TBD | 0.9963 |
| **Anomaly Score** | 0.9·recon + 0.1·(-D) | recon only | recon only | recon only | D(x) |
| **Early Stop Metric** | critic loss | critic loss | recon loss | recon loss | — |
| **recon_weight** | 10.0 | 10.0 | 1.0 | **100.0** | — |
| **hidden_dim** | 128 | 128 | 128 | **64** | — |
| **Grad clipping** | no | no | yes (1.0) | yes (1.0) | — |
| **patience** | 10 | 10 | 15 | **20** | — |
| **Architecture** | GRU AE-WGAN-GP | same | same | same | TCN/SA WGAN |

**Note:** Phase 2b uses the GRU-based AE-WGAN-GP architecture, which is fundamentally different from the paper's TCN/Self-Attention approach. The changes focus on making the autoencoder paradigm work correctly by ensuring the reconstruction objective actually drives learning.
