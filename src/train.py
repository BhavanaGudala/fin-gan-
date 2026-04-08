import yaml
import torch
import os
import sys
import numpy as np
from torch import optim
from tqdm import tqdm

# Resolve project root (one level above this src/ file)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # make all relative paths work from project root
sys.path.insert(0, os.path.join(ROOT, "src"))  # ensure src imports work

from dataset import get_loader
from model import Generator, Discriminator


def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1).to(device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    with torch.backends.cudnn.flags(enabled=False):
        d_interpolated = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.reshape(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# -------------------------
# Load Config
# -------------------------
cfg = yaml.safe_load(open("configs/config.yaml"))
device = torch.device(cfg["device"])

# -------------------------
# Load Dataset
# -------------------------
loader, ds = get_loader(
    cfg["train_csv"],
    cfg["seq_len"],
    cfg["label_column"],
    cfg["batch_size"],
    True,
    train_mode=True
)

feat_dim = ds.data.shape[1]
noise_dim = cfg.get("noise_dim", 32)
hidden_dim = cfg.get("hidden_dim", 64)
num_layers = cfg.get("num_layers", 2)
dropout = cfg.get("dropout", 0.2)
n_critic = cfg.get("n_critic", 5)
gp_lambda = cfg.get("gp_lambda", 10)
patience = cfg.get("patience", 10)
recon_weight = cfg.get("recon_weight", 1.0)

# Initialize Models
# Generator now uses feat_dim as input because it's an Autoencoder
G = Generator(feat_dim, hidden_dim, feat_dim, num_layers, dropout).to(device)
D = Discriminator(feat_dim, hidden_dim, num_layers, dropout).to(device)

# Optimizers — separate learning rates (slower critic to prevent divergence)
lr_G = cfg.get("lr_G", cfg.get("lr", 1e-4))
lr_D = cfg.get("lr_D", cfg.get("lr", 5e-5))
opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.9))

# LR scheduling — cosine annealing decays lr smoothly to break late-training plateaus
use_cosine = cfg.get("use_cosine_lr", False)
if use_cosine:
    T_max = cfg.get("cosine_T_max", cfg["epochs"])
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=T_max,
                                                     eta_min=cfg.get("cosine_eta_min_G", 1e-5))
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=T_max,
                                                     eta_min=cfg.get("cosine_eta_min_D", 5e-6))

# Create checkpoint folder
os.makedirs("checkpoints", exist_ok=True)

best_recon = float("inf")
patience_counter = 0


for epoch in range(cfg["epochs"]):

    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    epoch_recon_loss = 0.0

    for i, (x, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):

        x = x.to(device)
        b, seq_len, _ = x.shape

        # ==============================
        # Train Critic (n_critic steps)
        # ==============================
        # Feed real_x into the Autoencoder
        fake_x = G(x)

        real_score = D(x)
        fake_score = D(fake_x.detach())

        gp = gradient_penalty(D, x, fake_x.detach(), device)
        d_loss = -(torch.mean(real_score) - torch.mean(fake_score)) + gp_lambda * gp

        opt_D.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
        opt_D.step()

        epoch_d_loss += d_loss.item()

        # ==============================
        # Train Generator (every n_critic steps)
        # ==============================
        if (i + 1) % n_critic == 0:
            fake_x = G(x)
            
            # Generator wants to fool discriminator AND reconstruct the input
            critic_loss = -torch.mean(D(fake_x))
            recon_loss = torch.nn.functional.mse_loss(fake_x, x)
            g_loss = critic_loss + recon_weight * recon_loss

            opt_G.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()

            epoch_g_loss += g_loss.item()
            epoch_recon_loss += recon_loss.item()

    # End of Epoch
    avg_d_loss = epoch_d_loss / len(loader)
    g_updates = max(1, len(loader) // n_critic)
    avg_g_loss = epoch_g_loss / g_updates
    avg_recon = epoch_recon_loss / max(g_updates, 1)
    print(f"Epoch {epoch+1} - D: {avg_d_loss:.4f} | G: {avg_g_loss:.4f} | Recon: {avg_recon:.6f}")

    # Save best model (early stopping based on reconstruction loss)
    if avg_recon < best_recon:
        best_recon = avg_recon
        torch.save(D.state_dict(), "checkpoints/best_D.pth")
        torch.save(G.state_dict(), "checkpoints/best_G.pth")
        patience_counter = 0
        print(f"  -> Best model saved at epoch {epoch+1} (recon {best_recon:.6f})")
    else:
        patience_counter += 1
        print(f"  -> No improvement ({patience_counter}/{patience})")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    # Step LR schedulers
    if use_cosine:
        sched_G.step()
        sched_D.step()

# Save last model
torch.save(D.state_dict(), "checkpoints/D_last.pth")
# Also save as D.pth for inference compatibility
torch.save(D.state_dict(), "checkpoints/D.pth")

print("Training complete.")
