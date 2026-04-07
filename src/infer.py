import yaml, torch, pandas as pd
import os, sys

# Resolve project root (one level above this src/ file)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset import get_loader
from model import Discriminator, Generator

cfg = yaml.safe_load(open("configs/config.yaml"))
device = torch.device(cfg["device"])

hidden_dim = cfg.get("hidden_dim", 64)
num_layers = cfg.get("num_layers", 2)
dropout = cfg.get("dropout", 0.2)

loader, _ = get_loader(cfg["test_csv"], cfg["seq_len"],
                       cfg["label_column"], cfg["batch_size"],
                       False, train_mode=False)

feat_dim = loader.dataset.data.shape[1]
D = Discriminator(feat_dim, hidden_dim, num_layers, dropout).to(device)
D.load_state_dict(torch.load("checkpoints/best_D.pth", map_location=device))
D.eval()

G = Generator(feat_dim, hidden_dim, feat_dim, num_layers, dropout).to(device)
G.load_state_dict(torch.load("checkpoints/best_G.pth", map_location=device))
G.eval()

scores, labels = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        
        x_hat = G(x)
        # Reconstruction Error over the feature dimension (mean per timestep)
        recon_error = torch.mean((x - x_hat) ** 2, dim=-1)
        # Pool to sequence level
        recon_error = recon_error.mean(dim=1).cpu().numpy()
        
        # Critic Score (D gives high score for "real", low for "fake")
        # So -D(x) gives high score for anomalous data
        critic_score = -D(x).cpu().numpy().flatten()
        
        # Combined anomaly score: heavily weight reconstruction error
        # alpha can be tuned, e.g., 0.9.
        s = 0.9 * recon_error + 0.1 * critic_score
        
        scores.extend(s)
        labels.extend(y.numpy())

pd.DataFrame({"score": scores, "Label": labels}).to_csv(
    "checkpoints/inference_scores.csv", index=False)

print(f"Inference complete. Scored {len(scores)} samples.")
