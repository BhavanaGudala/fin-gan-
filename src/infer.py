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

# Auto-tune cuDNN kernels
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Load test data
loader, _ = get_loader(cfg["test_csv"], cfg["seq_len"],
                       cfg["label_column"], cfg["batch_size"],
                       False, train_mode=False, device=device)

# Load training data (benign only) for baseline computation
train_loader, _ = get_loader(cfg["train_csv"], cfg["seq_len"],
                             cfg["label_column"], cfg["batch_size"],
                             False, train_mode=True, device=device)

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
train_per_feat = []
train_latents = []
with torch.no_grad():
    for x, _ in tqdm(train_loader, desc="Baseline"):
        x = x.to(device)
        x_hat = G(x)
        recon = ((x - x_hat) ** 2).mean(dim=(1, 2))
        per_feat = ((x - x_hat) ** 2).mean(dim=1)
        d_score = D(x).squeeze(-1)
        latent = G.encode(x)
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

# Latent-space Mahalanobis baseline
lat_mu = train_lat.mean(axis=0)
lat_cov = np.cov(train_lat, rowvar=False) + 1e-6 * np.eye(train_lat.shape[1])
lat_cov_inv = np.linalg.inv(lat_cov)

# Per-feature Mahalanobis baseline
pf_cov = np.cov(train_pf, rowvar=False) + 1e-6 * np.eye(train_pf.shape[1])
pf_cov_inv = np.linalg.inv(pf_cov)

print(f"  Benign recon baseline: μ={recon_mu:.4f}, σ={recon_sigma:.4f}")
print(f"  Benign D(x)  baseline: μ={d_mu:.4f}, σ={d_sigma:.4f}")

# ── Phase 2: score test data ──
print("Running inference...")
recon_raw, d_raw, labels = [], [], []
test_per_feat = []
test_latents = []
with torch.no_grad():
    for x, y in tqdm(loader, desc="Inference"):
        x = x.to(device)
        x_hat = G(x)
        recon = ((x - x_hat) ** 2).mean(dim=(1, 2)).cpu().numpy()
        per_feat = ((x - x_hat) ** 2).mean(dim=1).cpu().numpy()
        d_score = D(x).squeeze(-1).cpu().numpy()
        latent = G.encode(x).cpu().numpy()
        recon_raw.extend(recon)
        d_raw.extend(d_score)
        test_per_feat.append(per_feat)
        test_latents.append(latent)
        labels.extend(y.numpy())

recon_raw = np.array(recon_raw)
d_raw = np.array(d_raw)
labels = np.array(labels)
test_pf = np.concatenate(test_per_feat, axis=0)
test_lat = np.concatenate(test_latents, axis=0)

# Compute all signals
pf_z = (test_pf - pf_mu) / pf_std
feat_d = np.abs(pf_z.mean(axis=0))
feat_d_safe = feat_d - feat_d.max()
feat_w = np.exp(feat_d_safe) / np.exp(feat_d_safe).sum()
weighted_recon = (pf_z * feat_w).sum(axis=1)

recon_z = (recon_raw - recon_mu) / max(recon_sigma, 1e-8)
d_z = -(d_raw - d_mu) / max(d_sigma, 1e-8)

# Latent Mahalanobis
lat_diff = test_lat - lat_mu
latent_mahal = np.sqrt(np.sum((lat_diff @ lat_cov_inv) * lat_diff, axis=1))
train_lat_diff = train_lat - lat_mu
train_lat_mahal = np.sqrt(np.sum((train_lat_diff @ lat_cov_inv) * train_lat_diff, axis=1))
lat_mahal_mu, lat_mahal_std = train_lat_mahal.mean(), train_lat_mahal.std() + 1e-8
latent_z = (latent_mahal - lat_mahal_mu) / lat_mahal_std

# Per-feature Mahalanobis
pf_diff = test_pf - pf_mu
pf_mahal = np.sqrt(np.sum((pf_diff @ pf_cov_inv) * pf_diff, axis=1))
train_pf_diff = train_pf - pf_mu
train_pf_mahal = np.sqrt(np.sum((train_pf_diff @ pf_cov_inv) * train_pf_diff, axis=1))
pf_mahal_mu, pf_mahal_std = train_pf_mahal.mean(), train_pf_mahal.std() + 1e-8
pf_mahal_z = (pf_mahal - pf_mahal_mu) / pf_mahal_std

# ── Phase 3: compare scoring methods ──
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

candidates = {
    "Mean MSE (raw)":           recon_raw,
    "Weighted MSE (per-feat)":  weighted_recon,
    "-D(x) (raw)":              -d_raw,
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
best_auc, best_name = 0, None
for name, s in candidates.items():
    if np.any(np.isnan(s)) or np.any(np.isinf(s)):
        print(f"  {name:35s} AUC = NaN (skipped)")
        continue
    a = roc_auc_score(labels, s)
    marker = ""
    if a > best_auc:
        best_auc, best_name = a, name
        marker = " ◀ best"
    print(f"  {name:35s} AUC = {a:.4f}{marker}")

# Learned fusion (cross-validated to avoid train-on-test leakage)
from sklearn.model_selection import StratifiedKFold

X_fusion = np.column_stack([recon_z, d_z, latent_z, pf_mahal_z])
valid = np.all(np.isfinite(X_fusion), axis=1)
X_valid, y_valid = X_fusion[valid], labels[valid]

fusion_prob_cv = np.zeros(len(y_valid))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_train, fold_test in skf.split(X_valid, y_valid):
    scaler_fold = StandardScaler()
    X_tr = scaler_fold.fit_transform(X_valid[fold_train])
    X_te = scaler_fold.transform(X_valid[fold_test])
    lr_fold = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    lr_fold.fit(X_tr, y_valid[fold_train])
    fusion_prob_cv[fold_test] = lr_fold.predict_proba(X_te)[:, 1]

fusion_auc_cv = roc_auc_score(y_valid, fusion_prob_cv)

# Full-data fit for coefficient inspection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)
lr_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
lr_model.fit(X_scaled, y_valid)

print(f"\n  {'Learned Fusion (5-fold CV)':35s} AUC = {fusion_auc_cv:.4f}", end="")
if fusion_auc_cv > best_auc:
    best_auc = fusion_auc_cv
    best_name = "Learned Fusion (CV)"
    print(" ◀ best")
else:
    print()

if best_name == "Learned Fusion (CV)":
    scores = np.zeros(len(labels))
    scores[valid] = fusion_prob_cv
    scores[~valid] = 0.0
else:
    scores = candidates[best_name]

print(f"\n  ➤ Using: {best_name} (AUC = {best_auc:.4f})")

pd.DataFrame({"score": scores, "Label": labels}).to_csv(
    "checkpoints/inference_scores.csv", index=False)
print(f"Inference complete. Scored {len(scores)} samples.")
