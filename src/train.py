import yaml
import torch
import os
import numpy as np
from torch import optim
from tqdm import tqdm
from dataset import get_loader
from model import Generator, Discriminator


def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1).to(device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

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

# Initialize Models
G = Generator(noise_dim, hidden_dim, feat_dim, num_layers, dropout).to(device)
D = Discriminator(feat_dim, hidden_dim, num_layers, dropout).to(device)

# Optimizers
opt_G = optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))

# Create checkpoint folder
os.makedirs("checkpoints", exist_ok=True)

best_loss = float("inf")
patience_counter = 0


for epoch in range(cfg["epochs"]):

    epoch_d_loss = 0.0
    epoch_g_loss = 0.0

    for i, (x, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):

        x = x.to(device)
        b, seq_len, _ = x.shape

        # ==============================
        # Train Critic (n_critic steps)
        # ==============================
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

        # ==============================
        # Train Generator (every n_critic steps)
        # ==============================
        if (i + 1) % n_critic == 0:
            z = torch.randn(b, seq_len, noise_dim).to(device)
            fake_x = G(z)
            g_loss = -torch.mean(D(fake_x))

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            epoch_g_loss += g_loss.item()

    # End of Epoch
    avg_d_loss = epoch_d_loss / len(loader)
    g_updates = max(1, len(loader) // n_critic)
    avg_g_loss = epoch_g_loss / g_updates
    print(f"Epoch {epoch+1} - Avg Critic Loss: {avg_d_loss:.6f} | Avg G Loss: {avg_g_loss:.6f}")

    # Save best model (early stopping based on critic loss)
    if avg_d_loss < best_loss:
        best_loss = avg_d_loss
        torch.save(D.state_dict(), "checkpoints/best_D.pth")
        torch.save(G.state_dict(), "checkpoints/best_G.pth")
        patience_counter = 0
        print(f"  -> Best model saved at epoch {epoch+1} (loss {best_loss:.6f})")
    else:
        patience_counter += 1
        print(f"  -> No improvement ({patience_counter}/{patience})")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Save last model
torch.save(D.state_dict(), "checkpoints/D_last.pth")
# Also save as D.pth for inference compatibility
torch.save(D.state_dict(), "checkpoints/D.pth")

print("Training complete.")
