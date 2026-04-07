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
    "seq_len": 10,
    "batch_size": 128,
    "epochs": 50,
    "lr": 1e-4,
    "noise_dim": 32,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "n_critic": 5,
    "gp_lambda": 10,
    "patience": 10,
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 3. Dataset

# %%
class FlowDataset(Dataset):
    """
    Network flow dataset with sliding window.
    - train_mode=True: keeps only benign rows, computes & saves normalization stats
    - train_mode=False: uses all rows, loads saved normalization stats
    """
    def __init__(self, csv_path, seq_len, label_col, train_mode=False, norm_path=None):
        df = pd.read_csv(csv_path)
        
        # Convert string labels to binary (handles both object and StringDtype)
        if pd.api.types.is_string_dtype(df[label_col]) or df[label_col].dtype == object:
            df[label_col] = (df[label_col] != "Benign").astype(int)
        
        if train_mode:
            df = df[df[label_col] == 0]
            if len(df) == 0:
                raise ValueError(f"No benign samples found in '{csv_path}'.")
        
        self.labels = df[label_col].values
        df = df.drop(columns=[label_col])
        
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        data = df.values.astype(np.float32)
        
        norm_file = norm_path or os.path.join(CKPT_DIR, "norm.npz")
        if train_mode:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-8
            np.savez(norm_file, mean=self.mean, std=self.std)
        else:
            stats = np.load(norm_file)
            self.mean, self.std = stats["mean"], stats["std"]
        
        self.data = (data - self.mean) / self.std
        self.seq_len = seq_len
        self.feature_names = list(df.columns)
    
    def __len__(self):
        return len(self.data) - self.seq_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y)


def get_loader(csv, seq_len, label, batch, shuffle, train_mode, norm_path=None):
    ds = FlowDataset(csv, seq_len, label, train_mode, norm_path)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True), ds

# %% [markdown]
# ## 4. Model Architecture
# 
# - **Generator:** 2-layer GRU + LayerNorm + Dropout → FC
# - **Discriminator:** 2-layer Bidirectional GRU + LayerNorm + Dropout → Attention Pooling → FC (no sigmoid)

# %%
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden, out_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(noise_dim, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, z):
        h, _ = self.gru(z)
        h = self.norm(h)
        h = self.dropout(h)
        return self.fc(h)


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
        self.fc = nn.Linear(hidden * 2, 1)

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
G = Generator(CONFIG["noise_dim"], CONFIG["hidden_dim"], feat_dim,
              CONFIG["num_layers"], CONFIG["dropout"]).to(device)
D = Discriminator(feat_dim, CONFIG["hidden_dim"],
                  CONFIG["num_layers"], CONFIG["dropout"]).to(device)

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
    # Disable CuDNN for this forward pass — CuDNN doesn't support
    # double backwards through RNNs, which gradient penalty requires
    with torch.backends.cudnn.flags(enabled=False):
        d_interpolated = D(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()



# %%
# Training loop
opt_G = optim.Adam(G.parameters(), lr=CONFIG["lr"], betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=CONFIG["lr"], betas=(0.5, 0.9))

n_critic = CONFIG["n_critic"]
gp_lambda = CONFIG["gp_lambda"]
patience = CONFIG["patience"]
noise_dim = CONFIG["noise_dim"]

# Metrics tracking
history = {
    "epoch": [], "d_loss": [], "g_loss": [],
    "d_real_mean": [], "d_fake_mean": [], "gp_mean": [],
    "epoch_time": []
}

best_loss = float("inf")
patience_counter = 0

print(f"Starting training for up to {CONFIG['epochs']} epochs...")
print(f"  Critic updates per generator update: {n_critic}")
print(f"  Early stopping patience: {patience}")
print(f"  Device: {device}\n")

for epoch in range(CONFIG["epochs"]):
    G.train()
    D.train()
    
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    epoch_d_real = 0.0
    epoch_d_fake = 0.0
    epoch_gp = 0.0
    g_steps = 0
    
    t_start = time.time()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    
    for i, (x, _) in pbar:
        x = x.to(device)
        b, seq_len, _ = x.shape
        
        # --- Train Critic ---
        z = torch.randn(b, seq_len, noise_dim).to(device)
        fake_x = G(z)
        
        real_score = D(x)
        fake_score = D(fake_x.detach())
        gp = gradient_penalty(D, x, fake_x.detach(), device)
        
        d_loss = -(torch.mean(real_score) - torch.mean(fake_score)) + gp_lambda * gp
        
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()
        
        epoch_d_loss += d_loss.item()
        epoch_d_real += real_score.mean().item()
        epoch_d_fake += fake_score.mean().item()
        epoch_gp += gp.item()
        
        # --- Train Generator (every n_critic steps) ---
        if (i + 1) % n_critic == 0:
            z = torch.randn(b, seq_len, noise_dim).to(device)
            fake_x = G(z)
            g_loss = -torch.mean(D(fake_x))
            
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            
            epoch_g_loss += g_loss.item()
            g_steps += 1
        
        pbar.set_postfix({
            "D": f"{d_loss.item():.4f}",
            "G": f"{epoch_g_loss / max(g_steps, 1):.4f}"
        })
    
    # Epoch stats
    n_batches = len(train_loader)
    avg_d = epoch_d_loss / n_batches
    avg_g = epoch_g_loss / max(g_steps, 1)
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
    history["epoch_time"].append(elapsed)
    
    print(f"  D_loss: {avg_d:.4f} | G_loss: {avg_g:.4f} | "
          f"D(real): {avg_real:.4f} | D(fake): {avg_fake:.4f} | "
          f"GP: {avg_gp:.4f} | Time: {elapsed:.1f}s")
    
    # Early stopping
    if avg_d < best_loss:
        best_loss = avg_d
        torch.save(D.state_dict(), os.path.join(CKPT_DIR, "best_D.pth"))
        torch.save(G.state_dict(), os.path.join(CKPT_DIR, "best_G.pth"))
        patience_counter = 0
        print(f"  ✓ Best model saved (loss: {best_loss:.6f})")
    else:
        patience_counter += 1
        print(f"  ✗ No improvement ({patience_counter}/{patience})")
    
    if patience_counter >= patience:
        print(f"\n⚡ Early stopping at epoch {epoch+1}")
        break

print(f"\nTraining complete! Best D loss: {best_loss:.6f}")
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
# ## 8. Inference — Score All Test Data

# %%
print("Loading test data (benign + attacks)...")
test_loader, test_ds = get_loader(
    CONFIG["test_csv"], CONFIG["seq_len"], CONFIG["label_column"],
    CONFIG["batch_size"], shuffle=False, train_mode=False
)

print(f"  Total test samples: {len(test_ds.data):,}")
print(f"  Test windows: {len(test_ds):,}")

# Load best model
D_best = Discriminator(feat_dim, CONFIG["hidden_dim"],
                       CONFIG["num_layers"], CONFIG["dropout"]).to(device)
D_best.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_D.pth"), map_location=device))
D_best.eval()

# Score all windows
scores = []
labels = []

print("Running inference...")
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Inference"):
        x = x.to(device)
        s = -D_best(x)  # anomaly score = negative critic score
        scores.extend(s.cpu().numpy().flatten())
        labels.extend(y.numpy())

scores = np.array(scores)
labels = np.array(labels)

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
  
  Best epoch:       {history['epoch'][np.argmin(history['d_loss'])]}
  Total train time: {sum(history['epoch_time']):.0f}s ({sum(history['epoch_time'])/60:.1f} min)
  
  Checkpoints saved to: {CKPT_DIR}/
""")
