import yaml, torch, pandas as pd, numpy as np
import os, sys
from tqdm import tqdm

# Resolve project root (one level above this src/ file)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset import get_loader
from model import Generator

cfg = yaml.safe_load(open("configs/config.yaml"))
device = torch.device(cfg["device"])

hidden_dim = cfg.get("hidden_dim", 64)
num_layers = cfg.get("num_layers", 2)
dropout = cfg.get("dropout", 0.2)
TOP_K = 10  # number of most-anomalous features to average

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

# ── Phase 1: per-feature error baselines on training (benign) data ──
# Mean-MSE hides attacks that differ in only a few features because the
# signal is diluted across all 77 features.  By computing the mean and
# std of per-feature reconstruction error on benign data, we can express
# test errors as z-scores.  Features that rarely have error on benign
# data will produce large z-scores even for small absolute deviations,
# catching subtle attacks like Syn floods.
print("Computing per-feature error baselines on training data...")
train_feat_errors = []
with torch.no_grad():
    for x, _ in tqdm(train_loader, desc="Baseline"):
        x = x.to(device)
        x_hat = G(x)
        fe = torch.mean((x - x_hat) ** 2, dim=1)  # (batch, features)
        train_feat_errors.append(fe.cpu())

train_feat_errors = torch.cat(train_feat_errors, dim=0)  # (N, features)
feat_mu = train_feat_errors.mean(dim=0)       # (features,)
feat_sigma = train_feat_errors.std(dim=0).clamp(min=1e-8)  # (features,)
np.savez("checkpoints/feat_baselines.npz",
         mu=feat_mu.numpy(), sigma=feat_sigma.numpy())
print(f"  Baselines saved ({feat_dim} features)")

# ── Phase 2: score test data with feature-standardised top-k ──
print("Running inference with feature-standardised scoring...")
feat_mu_d = feat_mu.to(device)
feat_sigma_d = feat_sigma.to(device)
scores, labels = [], []

with torch.no_grad():
    for x, y in tqdm(loader, desc="Inference"):
        x = x.to(device)
        x_hat = G(x)
        fe = torch.mean((x - x_hat) ** 2, dim=1)       # (batch, features)
        z = (fe - feat_mu_d) / feat_sigma_d              # z-scores
        topk, _ = torch.topk(z, k=TOP_K, dim=1)         # (batch, TOP_K)
        score = topk.mean(dim=1).cpu().numpy()            # (batch,)
        scores.extend(score)
        labels.extend(y.numpy())

pd.DataFrame({"score": scores, "Label": labels}).to_csv(
    "checkpoints/inference_scores.csv", index=False)

print(f"Inference complete. Scored {len(scores)} samples.")
