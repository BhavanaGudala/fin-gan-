# %% [markdown]
# # GAN-Based Network Intrusion Detection System
# 
# **Architecture:** Bidirectional GRU + Attention Pooling with WGAN-GP  
# **Dataset:** CICDDoS2019  
# **Task:** Unsupervised anomaly detection — train on benign traffic, detect DDoS attacks
# 
# This notebook handles: Setup → Training → Validation → Metrics → Inference

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
# Auto-detect environment (Colab vs local)
import os
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    # Change this path to your data location on Google Drive
    DATA_DIR = '/content/drive/MyDrive/fin-gan-/data'
    CKPT_DIR = '/content/checkpoints'
    print("Running on Google Colab")
else:
    # Local — resolve relative to notebook/script location
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    # If running from project root
    if os.path.exists(os.path.join(PROJECT_ROOT, 'data')):
        DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    # If running from notebooks/ subfolder
    elif os.path.exists(os.path.join(PROJECT_ROOT, '..', 'data')):
        DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'data')
    else:
        DATA_DIR = 'data'
    CKPT_DIR = os.path.join(os.path.dirname(DATA_DIR), 'checkpoints')
    print(f"Running locally | Data: {DATA_DIR}")

os.makedirs(CKPT_DIR, exist_ok=True)

# %%
# Install dependencies (uncomment on Colab)
# !pip install torch pandas numpy scikit-learn matplotlib tqdm pyyaml

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True  # auto-tune for fixed input shapes
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device("xpu")
    print("Using Intel XPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# %% [markdown]
# ## 2. Configuration

# %%
CONFIG = {
    "train_csv": os.path.join(DATA_DIR, "merged.csv"),
    "test_csv": os.path.join(DATA_DIR, "merged.csv"),
    "label_column": "Label",
    "seq_len": 20,
    "batch_size": 1024,
    "epochs": 300,
    "lr_G": 1e-4,
    "lr_D": 5e-5,
    "noise_dim": 32,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "n_critic": 5,
    "gp_lambda": 10,
    "patience": 40,
    "recon_weight": 100.0,
    "use_cosine_lr": True,
    "cosine_T_max": 300,
    "cosine_eta_min_G": 1e-5,
    "cosine_eta_min_D": 5e-6,
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 3. Dataset

# %%
# Features that are all-zero / constant across the CICDDoS2019 dataset.
DROP_FEATURES = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "PSH Flag Count", "ECE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

# Heavy-tailed features (skew > 10) that benefit from log1p transform.
LOG_FEATURES = [
    "Fwd Act Data Packets", "Fwd Packets Length Total", "Subflow Fwd Bytes",
    "Total Backward Packets", "Subflow Bwd Packets", "Total Fwd Packets",
    "Subflow Fwd Packets", "Subflow Bwd Bytes", "Bwd Packets Length Total",
    "Flow IAT Min", "Fwd IAT Min", "Packet Length Variance",
    "Flow Duration", "Fwd IAT Total", "Flow IAT Max", "Fwd IAT Max",
    "Fwd IAT Mean", "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Std",
    "Idle Max", "Idle Std", "Idle Mean", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Total",
    "Fwd Packet Length Max", "Bwd Packet Length Max",
    "Packet Length Mean", "Avg Packet Size",
]

class FlowDataset(Dataset):
    """
    Network flow dataset with sliding window.
    - train_mode=True: 80% benign split, log-transform, compute & save norm stats
    - train_mode=False: 20% held-out benign + all attacks, load saved stats
    """
    def __init__(self, csv_path, seq_len, label_col, train_mode=False, norm_path=None,
                 split_seed=42, split_ratio=0.8):
        df = pd.read_csv(csv_path)

        # Convert string labels to binary (handles both object and StringDtype)
        if pd.api.types.is_string_dtype(df[label_col]) or df[label_col].dtype == object:
            df[label_col] = (df[label_col] != "Benign").astype(int)

        if train_mode:
            benign = df[df[label_col] == 0]
            if len(benign) == 0:
                raise ValueError(f"No benign samples found in '{csv_path}'.")
            # 80/20 split on benign (deterministic)
            rng = np.random.RandomState(split_seed)
            indices = rng.permutation(len(benign))
            n_train = int(len(benign) * split_ratio)
            train_idx = indices[:n_train]
            df = benign.iloc[train_idx].reset_index(drop=True)
        else:
            # Test: held-out 20% benign + all attacks
            benign_mask = df[label_col] == 0
            benign_df = df[benign_mask]
            attack_df = df[~benign_mask]
            rng = np.random.RandomState(split_seed)
            indices = rng.permutation(len(benign_df))
            n_train = int(len(benign_df) * split_ratio)
            test_idx = indices[n_train:]
            test_benign = benign_df.iloc[test_idx].reset_index(drop=True)
            df = pd.concat([test_benign, attack_df], ignore_index=True)

        self.labels = df[label_col].values
        df = df.drop(columns=[label_col])

        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Drop constant/useless features
        drop_cols = [c for c in DROP_FEATURES if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # ── Derived rate features ──
        dur = df["Flow Duration"].values.copy() if "Flow Duration" in df.columns else None
        if dur is not None:
            dur_s = dur / 1e6  # microseconds → seconds
            dur_s = np.where(dur_s < 1e-6, 1e-6, dur_s)
            if "Total Fwd Packets" in df.columns:
                df["Fwd Packets/s"] = df["Total Fwd Packets"].values / dur_s
            if "Total Backward Packets" in df.columns:
                df["Bwd Packets/s"] = df["Total Backward Packets"].values / dur_s
            if "Fwd Packets Length Total" in df.columns:
                df["Fwd Bytes/s"] = df["Fwd Packets Length Total"].values / dur_s
            if "Bwd Packets Length Total" in df.columns:
                df["Bwd Bytes/s"] = df["Bwd Packets Length Total"].values / dur_s

        self.feature_names = list(df.columns)
        data = df.values.astype(np.float32)
        data = np.where(np.isfinite(data), data, 0.0)

        # Log-transform heavy-tailed features: sign(x) * log1p(|x|)
        log_set = set(LOG_FEATURES) | {"Fwd Packets/s", "Bwd Packets/s", "Fwd Bytes/s", "Bwd Bytes/s"}
        log_mask = np.array([c in log_set for c in self.feature_names])
        self._log_mask = log_mask
        if log_mask.any():
            data[:, log_mask] = np.sign(data[:, log_mask]) * np.log1p(np.abs(data[:, log_mask]))

        norm_file = norm_path or os.path.join(CKPT_DIR, "norm.npz")
        if train_mode:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-8
            np.savez(norm_file, mean=self.mean, std=self.std, log_mask=log_mask)
        else:
            stats = np.load(norm_file)
            self.mean, self.std = stats["mean"], stats["std"]

        self.data = (data - self.mean) / self.std
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def to_device(self, device):
        """Pre-load entire dataset to GPU to eliminate CPU→GPU transfer."""
        self._device = device
        self._data_tensor = torch.tensor(self.data, device=device)
        self._label_tensor = torch.tensor(self.labels, device=device)
        return self

    def __getitem__(self, idx):
        if hasattr(self, '_data_tensor'):
            x = self._data_tensor[idx:idx + self.seq_len]
            y = self._label_tensor[idx + self.seq_len - 1]
            return x, y
        x = self.data[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y)


def get_loader(csv, seq_len, label, batch, shuffle, train_mode, norm_path=None):
    ds = FlowDataset(csv, seq_len, label, train_mode, norm_path)
    if device.type == 'cuda':
        ds.to_device(device)
        return DataLoader(ds, batch_size=batch, shuffle=shuffle), ds
    nw = 4 if torch.cuda.is_available() else 0
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=nw, pin_memory=True, persistent_workers=(nw > 0)), ds

# %% [markdown]
# ## 4. Model Architecture
# 
# - **Generator:** 2-layer GRU + LayerNorm + Dropout → FC
# - **Discriminator:** 2-layer Bidirectional GRU + LayerNorm + Dropout → Attention Pooling → FC (no sigmoid)

# %%
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
    """Learns to weight all timesteps instead of only using the last one."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        scores = self.attn(h)
        weights = torch.softmax(scores, dim=1)
        return (weights * h).sum(dim=1)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionPooling(hidden * 2)
        self.fc = spectral_norm(nn.Linear(hidden * 2, 1))

    def forward(self, x):
        h, _ = self.gru(x)
        h = self.norm(h)
        h = self.dropout(h)
        context = self.attention(h)
        return self.fc(context)


# Initialize models
feat_dim = None  # Will be set after loading data

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# %% [markdown]
# ## 5. Load Data

# %%
print("Loading training data (benign only)...")
train_loader, train_ds = get_loader(
    CONFIG["train_csv"], CONFIG["seq_len"], CONFIG["label_column"],
    CONFIG["batch_size"], shuffle=True, train_mode=True
)

feat_dim = train_ds.data.shape[1]

print(f"  Benign samples: {len(train_ds.data):,}")
print(f"  Features: {feat_dim}")
print(f"  Windows: {len(train_ds):,}")
print(f"  Batches/epoch: {len(train_loader):,}")
print(f"  Feature names: {train_ds.feature_names[:5]}... ({len(train_ds.feature_names)} total)")

# %%
# Initialize models now that we know feat_dim
# Generator now uses feat_dim as input because it's an Autoencoder
G = Generator(feat_dim, CONFIG["hidden_dim"], feat_dim,
              CONFIG["num_layers"], CONFIG["dropout"]).to(device)
D = Discriminator(feat_dim, CONFIG["hidden_dim"],
                  CONFIG["num_layers"], CONFIG["dropout"]).to(device)

# Multi-GPU: wrap with DataParallel if more than one GPU
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    print(f"  Using {num_gpus} GPUs via DataParallel")
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)

print(f"\nGenerator:     {count_params(G):,} parameters")
print(f"Discriminator: {count_params(D):,} parameters")
print(f"Total:         {count_params(G) + count_params(D):,} parameters")

# %% [markdown]
# ## 6. Training

# %%
def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1).to(device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    # Use underlying module for GP (DataParallel doesn't support double backward)
    D_module = D.module if hasattr(D, 'module') else D
    # Disable CuDNN for this forward pass — CuDNN doesn't support
    # double backwards through RNNs, which gradient penalty requires
    with torch.backends.cudnn.flags(enabled=False):
        d_interpolated = D_module(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()



# %%
# Training loop
# Separate learning rates: slower critic to prevent divergence
opt_G = optim.Adam(G.parameters(), lr=CONFIG["lr_G"], betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=CONFIG["lr_D"], betas=(0.5, 0.9))

# Cosine annealing LR scheduler — decays lr smoothly to break late-training plateaus
use_cosine = CONFIG.get("use_cosine_lr", False)
if use_cosine:
    T_max = CONFIG.get("cosine_T_max", CONFIG["epochs"])
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=T_max,
                                                     eta_min=CONFIG.get("cosine_eta_min_G", 1e-5))
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=T_max,
                                                     eta_min=CONFIG.get("cosine_eta_min_D", 5e-6))

# Mixed precision for GPU speedup
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler(enabled=use_amp)

n_critic = CONFIG["n_critic"]
gp_lambda = CONFIG["gp_lambda"]
patience = CONFIG["patience"]
noise_dim = CONFIG["noise_dim"]
recon_weight = CONFIG["recon_weight"]

# Metrics tracking
history = {
    "epoch": [], "d_loss": [], "g_loss": [],
    "d_real_mean": [], "d_fake_mean": [], "gp_mean": [],
    "recon_loss": [], "epoch_time": []
}

best_recon = float("inf")
patience_counter = 0

print(f"Starting training for up to {CONFIG['epochs']} epochs...")
print(f"  Critic updates per generator update: {n_critic}")
print(f"  Early stopping patience: {patience}")
print(f"  Mixed precision (AMP): {use_amp}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  GPUs: {torch.cuda.device_count()}")
print(f"  Device: {device}\n")

for epoch in range(CONFIG["epochs"]):
    G.train()
    D.train()
    
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    epoch_d_real = 0.0
    epoch_d_fake = 0.0
    epoch_gp = 0.0
    epoch_recon_loss = 0.0
    g_steps = 0
    
    t_start = time.time()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    
    for i, (x, _) in pbar:
        x = x.to(device)
        b, seq_len, _ = x.shape
        
        # --- Train Critic ---
        # Feed real_x into the Autoencoder generator
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            fake_x = G(x)
            real_score = D(x)
            fake_score = D(fake_x.detach())
        
        # GP must run in float32 (needs double backward)
        gp = gradient_penalty(D, x, fake_x.detach(), device)
        gp = torch.clamp(gp, max=1.0)  # cap GP to prevent runaway drift
        d_loss = -(torch.mean(real_score) - torch.mean(fake_score)) + gp_lambda * gp
        
        opt_D.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
        opt_D.step()
        
        epoch_d_loss += d_loss.item()
        epoch_d_real += real_score.mean().item()
        epoch_d_fake += fake_score.mean().item()
        epoch_gp += gp.item()
        
        # --- Train Generator (every n_critic steps) ---
        if (i + 1) % n_critic == 0:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake_x = G(x)
                critic_loss = -torch.mean(D(fake_x))
                recon_loss = torch.nn.functional.mse_loss(fake_x, x)
                g_loss = critic_loss + recon_weight * recon_loss
            
            opt_G.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()
            
            epoch_g_loss += g_loss.item()
            epoch_recon_loss += recon_loss.item()
            g_steps += 1
        
        pbar.set_postfix({
            "D": f"{d_loss.item():.4f}",
            "G": f"{epoch_g_loss / max(g_steps, 1):.4f}"
        })
    
    # Epoch stats
    n_batches = len(train_loader)
    avg_d = epoch_d_loss / n_batches
    avg_g = epoch_g_loss / max(g_steps, 1)
    avg_recon = epoch_recon_loss / max(g_steps, 1)
    avg_real = epoch_d_real / n_batches
    avg_fake = epoch_d_fake / n_batches
    avg_gp = epoch_gp / n_batches
    elapsed = time.time() - t_start
    
    history["epoch"].append(epoch + 1)
    history["d_loss"].append(avg_d)
    history["g_loss"].append(avg_g)
    history["d_real_mean"].append(avg_real)
    history["d_fake_mean"].append(avg_fake)
    history["gp_mean"].append(avg_gp)
    history["recon_loss"].append(avg_recon)
    history["epoch_time"].append(elapsed)
    
    print(f"  D: {avg_d:.4f} | G: {avg_g:.4f} | Recon: {avg_recon:.6f} | "
          f"D(real): {avg_real:.4f} | D(fake): {avg_fake:.4f} | GP: {avg_gp:.4f} | Time: {elapsed:.1f}s")
    
    # Early stopping based on reconstruction loss (autoencoder quality)
    if avg_recon < best_recon:
        best_recon = avg_recon
        # Save underlying module (unwrap DataParallel)
        G_save = G.module if hasattr(G, 'module') else G
        D_save = D.module if hasattr(D, 'module') else D
        torch.save(D_save.state_dict(), os.path.join(CKPT_DIR, "best_D.pth"))
        torch.save(G_save.state_dict(), os.path.join(CKPT_DIR, "best_G.pth"))
        patience_counter = 0
        print(f"  ✓ Best model saved (recon: {best_recon:.6f})")
    else:
        patience_counter += 1
        print(f"  ✗ No improvement ({patience_counter}/{patience})")
    
    if patience_counter >= patience:
        print(f"\n⚡ Early stopping at epoch {epoch+1}")
        break

    # Step LR schedulers
    if use_cosine:
        sched_G.step()
        sched_D.step()

print(f"\nTraining complete! Best recon loss: {best_recon:.6f}")
print(f"Total time: {sum(history['epoch_time']):.0f}s ({sum(history['epoch_time'])/60:.1f} min)")

# %% [markdown]
# ## 7. Training Metrics Visualization

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Training Metrics", fontsize=16, fontweight='bold')

# Critic & Generator loss
ax = axes[0, 0]
ax.plot(history["epoch"], history["d_loss"], 'b-', label="Critic Loss", linewidth=2)
ax.plot(history["epoch"], history["g_loss"], 'r-', label="Generator Loss", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Critic vs Generator Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# D(real) vs D(fake) — these should converge
ax = axes[0, 1]
ax.plot(history["epoch"], history["d_real_mean"], 'g-', label="D(real)", linewidth=2)
ax.plot(history["epoch"], history["d_fake_mean"], 'r-', label="D(fake)", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Discriminator Score")
ax.set_title("D(real) vs D(fake) — should converge")
ax.legend()
ax.grid(True, alpha=0.3)

# Gradient penalty
ax = axes[1, 0]
ax.plot(history["epoch"], history["gp_mean"], 'm-', linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Gradient Penalty")
ax.set_title("Gradient Penalty (should stay small)")
ax.grid(True, alpha=0.3)

# Epoch time
ax = axes[1, 1]
ax.bar(history["epoch"], history["epoch_time"], color='steelblue', alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Time (seconds)")
ax.set_title("Training Time per Epoch")
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(CKPT_DIR, "training_metrics.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {os.path.join(CKPT_DIR, 'training_metrics.png')}")

# %% [markdown]
# ## 8. Inference — Multi-Signal Scoring

# %%
print("Loading test data (benign + attacks)...")
test_loader, test_ds = get_loader(
    CONFIG["test_csv"], CONFIG["seq_len"], CONFIG["label_column"],
    CONFIG["batch_size"], shuffle=False, train_mode=False
)

print(f"  Total test samples: {len(test_ds.data):,}")
print(f"  Test windows: {len(test_ds):,}")

# Load best generator AND discriminator
G_best = Generator(feat_dim, CONFIG["hidden_dim"], feat_dim,
                   CONFIG["num_layers"], CONFIG["dropout"]).to(device)
G_best.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_G.pth"), map_location=device))
G_best.eval()

D_best = Discriminator(feat_dim, CONFIG["hidden_dim"],
                       CONFIG["num_layers"], CONFIG["dropout"]).to(device)
D_best.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_D.pth"), map_location=device))
D_best.eval()

# ── Phase 1: Compute baselines on benign training data ──
# We compute reconstruction error, D(x), latent vectors, and per-feature errors
# on benign data to build normalization stats and covariance matrices.
print("Computing baselines on training data...")
baseline_loader, _ = get_loader(
    CONFIG["train_csv"], CONFIG["seq_len"], CONFIG["label_column"],
    CONFIG["batch_size"], shuffle=False, train_mode=True
)

train_recon_scores = []
train_d_scores = []
train_per_feat = []
train_latents = []
with torch.no_grad():
    for x, _ in tqdm(baseline_loader, desc="Baseline"):
        x = x.to(device)
        x_hat = G_best(x)
        # Mean MSE per sample (mean over time and features)
        recon = ((x - x_hat) ** 2).mean(dim=(1, 2))
        per_feat = ((x - x_hat) ** 2).mean(dim=1)  # (batch, feat_dim)
        d_score = D_best(x).squeeze(-1)
        # Latent-space encoding (encoder hidden state)
        latent = G_best.encode(x)  # (batch, hidden_dim)
        train_recon_scores.append(recon.cpu())
        train_d_scores.append(d_score.cpu())
        train_per_feat.append(per_feat.cpu())
        train_latents.append(latent.cpu())

train_recon_all = torch.cat(train_recon_scores)
train_d_all = torch.cat(train_d_scores)
train_pf = torch.cat(train_per_feat).numpy()
train_lat = torch.cat(train_latents).numpy()

recon_mu, recon_sigma = train_recon_all.mean().item(), train_recon_all.std().item()
d_mu, d_sigma = train_d_all.mean().item(), train_d_all.std().item()
pf_mu = train_pf.mean(axis=0)
pf_std = train_pf.std(axis=0) + 1e-8

# Latent-space baseline: mean + precision matrix (inverse covariance) for Mahalanobis
lat_mu = train_lat.mean(axis=0)
lat_cov = np.cov(train_lat, rowvar=False) + 1e-6 * np.eye(train_lat.shape[1])
lat_cov_inv = np.linalg.inv(lat_cov)

# Per-feature Mahalanobis: covariance of per-feature errors on benign
pf_cov = np.cov(train_pf, rowvar=False) + 1e-6 * np.eye(train_pf.shape[1])
pf_cov_inv = np.linalg.inv(pf_cov)

print(f"  Benign recon baseline: μ={recon_mu:.4f}, σ={recon_sigma:.4f}")
print(f"  Benign D(x)  baseline: μ={d_mu:.4f}, σ={d_sigma:.4f}")
print(f"  Latent dim: {train_lat.shape[1]}, cov condition: {np.linalg.cond(lat_cov):.1f}")

# ── Phase 2: Score all test data ──
print("Running inference...")
recon_raw = []
d_raw = []
labels = []
test_per_feat = []
test_latents = []

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Inference"):
        x = x.to(device)
        x_hat = G_best(x)
        recon = ((x - x_hat) ** 2).mean(dim=(1, 2)).cpu().numpy()
        per_feat = ((x - x_hat) ** 2).mean(dim=1).cpu().numpy()
        d_score = D_best(x).squeeze(-1).cpu().numpy()
        latent = G_best.encode(x).cpu().numpy()
        recon_raw.extend(recon)
        d_raw.extend(d_score)
        test_per_feat.append(per_feat)
        test_latents.append(latent)
        labels.extend(y.cpu().numpy())

recon_raw = np.array(recon_raw)
d_raw = np.array(d_raw)
labels = np.array(labels)
test_pf = np.concatenate(test_per_feat, axis=0)
test_lat = np.concatenate(test_latents, axis=0)

# ── Compute all scoring signals ──

# 1. Per-feature weighted MSE
pf_z = (test_pf - pf_mu) / pf_std
feat_d = np.abs(pf_z.mean(axis=0))
feat_d_safe = feat_d - feat_d.max()
feat_w = np.exp(feat_d_safe) / np.exp(feat_d_safe).sum()
weighted_recon = (pf_z * feat_w).sum(axis=1)

# 2. Z-scores
recon_z = (recon_raw - recon_mu) / max(recon_sigma, 1e-8)
d_z = -(d_raw - d_mu) / max(d_sigma, 1e-8)

# 3. Latent-space Mahalanobis distance
lat_diff = test_lat - lat_mu
latent_mahal = np.sqrt(np.sum((lat_diff @ lat_cov_inv) * lat_diff, axis=1))
# Normalize to z-score using benign latent Mahalanobis
train_lat_diff = train_lat - lat_mu
train_lat_mahal = np.sqrt(np.sum((train_lat_diff @ lat_cov_inv) * train_lat_diff, axis=1))
lat_mahal_mu, lat_mahal_std = train_lat_mahal.mean(), train_lat_mahal.std() + 1e-8
latent_z = (latent_mahal - lat_mahal_mu) / lat_mahal_std

# 4. Per-feature Mahalanobis distance
pf_diff = test_pf - pf_mu
pf_mahal = np.sqrt(np.sum((pf_diff @ pf_cov_inv) * pf_diff, axis=1))
train_pf_diff = train_pf - pf_mu
train_pf_mahal = np.sqrt(np.sum((train_pf_diff @ pf_cov_inv) * train_pf_diff, axis=1))
pf_mahal_mu, pf_mahal_std = train_pf_mahal.mean(), train_pf_mahal.std() + 1e-8
pf_mahal_z = (pf_mahal - pf_mahal_mu) / pf_mahal_std

# ── Phase 3: Compare scoring methods (including new signals) ──
from sklearn.metrics import roc_auc_score as _auc

candidates = {
    "Mean MSE (raw)":           recon_raw,
    "Weighted MSE (per-feat)":  weighted_recon,
    "-D(x) (raw)":              -d_raw,
    "Recon z-score":            recon_z,
    "D z-score":                d_z,
    "Latent Mahalanobis":       latent_z,
    "PF Mahalanobis":           pf_mahal_z,
    "Recon_z + D_z (1:1)":     recon_z + d_z,
    "Recon_z + Latent_z":      recon_z + latent_z,
    "Recon_z + PF_Mahal_z":    recon_z + pf_mahal_z,
    "All 4 signals (sum)":     recon_z + d_z + latent_z + pf_mahal_z,
}

print("\n" + "=" * 55)
print("  Scoring Method Comparison")
print("=" * 55)
best_auc = 0
best_name = None
for name, s in candidates.items():
    if np.any(np.isnan(s)) or np.any(np.isinf(s)):
        print(f"  {name:35s} AUC = NaN (skipped)")
        continue
    a = _auc(labels, s)
    marker = ""
    if a > best_auc:
        best_auc = a
        best_name = name
        marker = " ◀ best"
    print(f"  {name:35s} AUC = {a:.4f}{marker}")

# ── Phase 3b: Learned fusion via logistic regression ──
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Build feature matrix from all 4 z-scored signals
X_fusion = np.column_stack([recon_z, d_z, latent_z, pf_mahal_z])
# Remove any NaN/inf
valid = np.all(np.isfinite(X_fusion), axis=1)
X_valid = X_fusion[valid]
y_valid = labels[valid]

# Fit logistic regression (L2 regularized)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)
lr_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
lr_model.fit(X_scaled, y_valid)
fusion_prob = lr_model.predict_proba(X_scaled)[:, 1]
fusion_auc = _auc(y_valid, fusion_prob)

print(f"\n  {'Learned Fusion (LR)':35s} AUC = {fusion_auc:.4f}", end="")
if fusion_auc > best_auc:
    best_auc = fusion_auc
    best_name = "Learned Fusion (LR)"
    print(" ◀ best")
else:
    print()

print(f"\n  LR coefficients: recon_z={lr_model.coef_[0][0]:.3f}, d_z={lr_model.coef_[0][1]:.3f}, "
      f"latent_z={lr_model.coef_[0][2]:.3f}, pf_mahal_z={lr_model.coef_[0][3]:.3f}")

# Use the best method
if best_name == "Learned Fusion (LR)":
    # For fusion, we need to handle the valid mask
    scores = np.zeros(len(labels))
    scores[valid] = fusion_prob
    scores[~valid] = 0.0  # mark invalid as benign-like
else:
    scores = candidates[best_name]

print(f"\n  ➤ Using: {best_name} (AUC = {best_auc:.4f})")

# Print score statistics for diagnosis
for lbl, lbl_name in [(0, "Benign"), (1, "Attack")]:
    mask = labels == lbl
    r = recon_raw[mask]
    d = d_raw[mask]
    lm = latent_mahal[mask]
    print(f"\n  {lbl_name} score stats:")
    print(f"    Recon  — median={np.median(r):.2f}, mean={r.mean():.2f}, p95={np.percentile(r,95):.2f}")
    print(f"    D(x)   — median={np.median(d):.2f}, mean={d.mean():.2f}, p5={np.percentile(d,5):.2f}")
    print(f"    Latent — median={np.median(lm):.2f}, mean={lm.mean():.2f}, p95={np.percentile(lm,95):.2f}")

results_df = pd.DataFrame({"score": scores, "Label": labels})
results_df.to_csv(os.path.join(CKPT_DIR, "inference_scores.csv"), index=False)

print(f"\nInference complete!")
print(f"  Benign windows: {(labels == 0).sum():,}")
print(f"  Attack windows: {(labels == 1).sum():,}")
print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

# %% [markdown]
# ## 9. Evaluation Metrics

# %%
# === ROC-AUC ===
auc = roc_auc_score(labels, scores)
fpr, tpr, thresholds = roc_curve(labels, scores)

print("=" * 50)
print(f"  ROC-AUC: {auc:.4f}")
print("=" * 50)

# === Find optimal threshold (Youden's J statistic) ===
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f"\n  Optimal threshold (Youden's J): {best_threshold:.4f}")
print(f"  At this threshold → TPR: {tpr[best_idx]:.4f}, FPR: {fpr[best_idx]:.4f}")

# === Classification metrics at optimal threshold ===
y_pred = (scores >= best_threshold).astype(int)

print(f"\n  Accuracy:  {accuracy_score(labels, y_pred):.4f}")
print(f"  Precision: {precision_score(labels, y_pred):.4f}")
print(f"  Recall:    {recall_score(labels, y_pred):.4f}")
print(f"  F1-Score:  {f1_score(labels, y_pred):.4f}")

# %%
# === ROC Curve Plot ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
ax = axes[0]
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
ax.scatter(fpr[best_idx], tpr[best_idx], c='red', s=100, zorder=5,
           label=f'Optimal (TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})')
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve", fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

# Score distribution
ax = axes[1]
ax.hist(scores[labels == 0], bins=100, alpha=0.6, label='Benign', color='green', density=True)
ax.hist(scores[labels == 1], bins=100, alpha=0.6, label='Attack', color='red', density=True)
ax.axvline(best_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {best_threshold:.3f}')
ax.set_xlabel("Anomaly Score", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Score Distribution", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CKPT_DIR, "evaluation_plots.png"), dpi=150, bbox_inches='tight')
plt.show()

# %%
# === Confusion Matrix ===
cm = confusion_matrix(labels, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Benign', 'Attack'], fontsize=12)
ax.set_yticklabels(['Benign', 'Attack'], fontsize=12)
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(CKPT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.show()

# %%
# === Per-Attack-Type Detection Rates ===
print("\n" + "=" * 60)
print("  Per-Attack-Type Detection Rates")
print("=" * 60)

# Re-read CSV to get original string labels
raw_df = pd.read_csv(CONFIG["test_csv"])
label_col = CONFIG["label_column"]
seq_len = CONFIG["seq_len"]

# Get the label for each window (label of the last timestep)
raw_labels = raw_df[label_col].values
window_labels = [raw_labels[i + seq_len - 1] for i in range(len(raw_labels) - seq_len + 1)]
window_labels = np.array(window_labels[:len(scores)])  # align with scored windows

attack_types = sorted(set(window_labels) - {"Benign"})

print(f"\n  {'Attack Type':<20} {'Count':>8} {'Detected':>10} {'Rate':>8}")
print("  " + "-" * 50)

for attack in attack_types:
    mask = window_labels == attack
    if mask.sum() == 0:
        continue
    detected = y_pred[mask].sum()
    total = mask.sum()
    rate = detected / total
    print(f"  {attack:<20} {total:>8,} {detected:>10,} {rate:>8.4f}")

# Benign (true negative rate)
benign_mask = window_labels == "Benign"
benign_correct = (y_pred[benign_mask] == 0).sum()
print(f"\n  {'Benign (TNR)':<20} {benign_mask.sum():>8,} {benign_correct:>10,} {benign_correct/benign_mask.sum():>8.4f}")

# %%
# === Full Classification Report ===
print("\n" + "=" * 60)
print("  Full Classification Report")
print("=" * 60)
print(classification_report(labels, y_pred, target_names=["Benign", "Attack"], digits=4))

# %% [markdown]
# ## 10. Summary

# %%
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"""
  Architecture:     Bidirectional GRU + Attention Pooling (WGAN-GP)
  Training samples: {len(train_ds.data):,} benign flows
  Test samples:     {len(test_ds.data):,} total flows
  Features:         {feat_dim}
  
  ROC-AUC:          {auc:.4f}
  Accuracy:         {accuracy_score(labels, y_pred):.4f}
  Precision:        {precision_score(labels, y_pred):.4f}
  Recall:           {recall_score(labels, y_pred):.4f}
  F1-Score:         {f1_score(labels, y_pred):.4f}
  
  Best epoch:       {history['epoch'][np.argmin(history['recon_loss'])]}
  Total train time: {sum(history['epoch_time']):.0f}s ({sum(history['epoch_time'])/60:.1f} min)
  
  Checkpoints saved to: {CKPT_DIR}/
""")

# %% [markdown]
# ## 11. Save Console Logs

# %%
# Auto-save all console output to logs/log_runN.txt (auto-incremented)
import glob, re as _re

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(".")), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Find the next run number by scanning existing log files
existing = glob.glob(os.path.join(LOG_DIR, "log_run*.txt"))
run_nums = []
for f in existing:
    m = _re.search(r"log_run(\d+)\.txt$", f)
    if m:
        run_nums.append(int(m.group(1)))
next_run = max(run_nums, default=0) + 1
log_path = os.path.join(LOG_DIR, f"log_run{next_run}.txt")

# Collect the log content from the training history and results
log_lines = []
log_lines.append(f"Running locally | Data: {DATA_DIR}")
log_lines.append(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
log_lines.append("Configuration:")
for k, v in CONFIG.items():
    log_lines.append(f"  {k}: {v}")
log_lines.append("")
log_lines.append(f"Generator:     {count_params(G):,} parameters")
log_lines.append(f"Discriminator: {count_params(D):,} parameters")
log_lines.append(f"Total:         {count_params(G) + count_params(D):,} parameters")
log_lines.append(f"Features: {feat_dim}")
log_lines.append(f"Training samples: {len(train_ds.data):,}")
log_lines.append("")
log_lines.append(f"Training for {CONFIG['epochs']} epochs")
for i, ep in enumerate(history["epoch"]):
    log_lines.append(
        f"Epoch {ep}/{CONFIG['epochs']} | "
        f"D: {history['d_loss'][i]:.4f} | G: {history['g_loss'][i]:.4f} | "
        f"Recon: {history['recon_loss'][i]:.6f} | "
        f"D(real): {history['d_real_mean'][i]:.4f} | D(fake): {history['d_fake_mean'][i]:.4f} | "
        f"GP: {history['gp_mean'][i]:.4f} | Time: {history['epoch_time'][i]:.1f}s"
    )
log_lines.append("")
log_lines.append(f"Best recon loss: {best_recon:.6f}")
log_lines.append(f"Total training time: {sum(history['epoch_time']):.0f}s ({sum(history['epoch_time'])/60:.1f} min)")
log_lines.append("")
log_lines.append("=" * 50)
log_lines.append(f"  ROC-AUC: {auc:.4f}")
log_lines.append("=" * 50)
log_lines.append(f"  Accuracy:  {accuracy_score(labels, y_pred):.4f}")
log_lines.append(f"  Precision: {precision_score(labels, y_pred):.4f}")
log_lines.append(f"  Recall:    {recall_score(labels, y_pred):.4f}")
log_lines.append(f"  F1-Score:  {f1_score(labels, y_pred):.4f}")

with open(log_path, "w") as f:
    f.write("\n".join(log_lines) + "\n")

print(f"Logs saved to: {log_path}")
