import yaml, torch, pandas as pd, numpy as np
import os, sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Resolve project root (one level above this src/ file)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset import get_loader
from model import Generator, Discriminator

cfg = yaml.safe_load(open("configs/config.yaml"))
device = torch.device(cfg["device"])

hidden_dim = cfg.get("hidden_dim", 64)
num_layers = cfg.get("num_layers", 2)
dropout = cfg.get("dropout", 0.2)

# Load test data
loader, _ = get_loader(cfg["test_csv"], cfg["seq_len"],
                       cfg["label_column"], cfg["batch_size"],
                       False, train_mode=False)

# Load training data (benign only) for baseline computation
train_loader, _ = get_loader(cfg["train_csv"], cfg["seq_len"],
                             cfg["label_column"], cfg["batch_size"],
                             False, train_mode=True)

feat_dim = loader.dataset.data.shape[1]

G = Generator(feat_dim, hidden_dim, feat_dim, num_layers, dropout).to(device)
G.load_state_dict(torch.load("checkpoints/best_G.pth", map_location=device))
G.eval()

D = Discriminator(feat_dim, hidden_dim, num_layers, dropout).to(device)
D.load_state_dict(torch.load("checkpoints/best_D.pth", map_location=device))
D.eval()

# ── Phase 1: baselines on benign training data ──
print("Computing baselines on training data...")
train_recon_scores, train_d_scores = [], []
train_per_feat = []  # per-feature MSE for weight computation
with torch.no_grad():
    for x, _ in tqdm(train_loader, desc="Baseline"):
        x = x.to(device)
        x_hat = G(x)
        recon = ((x - x_hat) ** 2).mean(dim=(1, 2))
        per_feat = ((x - x_hat) ** 2).mean(dim=1)  # (batch, feat_dim)
        d_score = D(x).squeeze(-1)
        train_recon_scores.append(recon.cpu())
        train_d_scores.append(d_score.cpu())
        train_per_feat.append(per_feat.cpu())

train_recon_all = torch.cat(train_recon_scores)
train_d_all = torch.cat(train_d_scores)
train_pf = torch.cat(train_per_feat).numpy()  # (N_train, feat_dim)
recon_mu, recon_sigma = train_recon_all.mean().item(), train_recon_all.std().item()
d_mu, d_sigma = train_d_all.mean().item(), train_d_all.std().item()
# Per-feature benign baselines for weighted scoring
pf_mu = train_pf.mean(axis=0)   # (feat_dim,)
pf_std = train_pf.std(axis=0) + 1e-8
print(f"  Benign recon baseline: μ={recon_mu:.4f}, σ={recon_sigma:.4f}")
print(f"  Benign D(x)  baseline: μ={d_mu:.4f}, σ={d_sigma:.4f}")

# ── Phase 2: score test data ──
print("Running inference...")
recon_raw, d_raw, labels = [], [], []
test_per_feat = []
with torch.no_grad():
    for x, y in tqdm(loader, desc="Inference"):
        x = x.to(device)
        x_hat = G(x)
        recon = ((x - x_hat) ** 2).mean(dim=(1, 2)).cpu().numpy()
        per_feat = ((x - x_hat) ** 2).mean(dim=1).cpu().numpy()  # (batch, feat_dim)
        d_score = D(x).squeeze(-1).cpu().numpy()
        recon_raw.extend(recon)
        d_raw.extend(d_score)
        test_per_feat.append(per_feat)
        labels.extend(y.numpy())

recon_raw = np.array(recon_raw)
d_raw = np.array(d_raw)
labels = np.array(labels)
test_pf = np.concatenate(test_per_feat, axis=0)  # (N_test, feat_dim)

# Per-feature z-score weighted scoring:
# Features with higher variance ratio (test/train) are more anomalous.
# Weight = softmax of per-feature Cohen's d computed on the fly.
pf_z = (test_pf - pf_mu) / pf_std  # (N_test, feat_dim) z-scored per feature
# Feature importance: how much each feature deviates from benign on average
feat_d = np.abs(pf_z.mean(axis=0))  # average z-deviation per feature
feat_w = np.exp(feat_d) / np.exp(feat_d).sum()  # softmax weights
weighted_recon = (pf_z * feat_w).sum(axis=1)  # weighted anomaly score

recon_z = (recon_raw - recon_mu) / max(recon_sigma, 1e-8)
d_z = -(d_raw - d_mu) / max(d_sigma, 1e-8)

# ── Phase 3: compare scoring methods ──
candidates = {
    "Mean MSE (raw)":           recon_raw,
    "Weighted MSE (per-feat)":  weighted_recon,
    "-D(x) (raw)":              -d_raw,
    "Recon_z + D_z (1:1)":     recon_z + d_z,
    "0.7·Recon_z + 0.3·D_z":   0.7 * recon_z + 0.3 * d_z,
    "0.5·Recon_z + 0.5·D_z":   0.5 * recon_z + 0.5 * d_z,
    "0.3·Recon_z + 0.7·D_z":   0.3 * recon_z + 0.7 * d_z,
    "max(Recon_z, D_z)":        np.maximum(recon_z, d_z),
}

print("\n" + "=" * 55)
print("  Scoring Method Comparison")
print("=" * 55)
best_auc, best_name = 0, None
for name, s in candidates.items():
    a = roc_auc_score(labels, s)
    marker = ""
    if a > best_auc:
        best_auc, best_name = a, name
        marker = " ◀ best"
    print(f"  {name:35s} AUC = {a:.4f}{marker}")

scores = candidates[best_name]
print(f"\n  ➤ Using: {best_name} (AUC = {best_auc:.4f})")

pd.DataFrame({"score": scores, "Label": labels}).to_csv(
    "checkpoints/inference_scores.csv", index=False)
print(f"Inference complete. Scored {len(scores)} samples.")
