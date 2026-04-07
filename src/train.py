import yaml
import torch
import os
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

    # reshape instead of view (fixes runtime error)
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
noise_dim = 32

# Initialize Models
G = Generator(noise_dim, 64, feat_dim).to(device)
D = Discriminator(feat_dim, 64).to(device)

# Optimizers
opt_G = optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))

# Create checkpoint folder
os.makedirs("checkpoints", exist_ok=True)

best_loss = float("inf")


for epoch in range(cfg["epochs"]):

    epoch_d_loss = 0.0

    for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}"):

        x = x.to(device)
        b, seq_len, _ = x.shape

        # Generate Fake Sequences
        z = torch.randn(b, seq_len, noise_dim).to(device)
        fake_x = G(z)

        # Train Critic (Discriminator)
        real_score = D(x)
        fake_score = D(fake_x.detach())

        gp = gradient_penalty(D, x, fake_x.detach(), device)

        d_loss = -(torch.mean(real_score) - torch.mean(fake_score)) + 10 * gp

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        epoch_d_loss += d_loss.item()

        # Train Generator
        g_loss = -torch.mean(D(fake_x))

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    # End of Epoch
    avg_d_loss = epoch_d_loss / len(loader)
    print(f"Epoch {epoch+1} - Avg Critic Loss: {avg_d_loss:.6f}")

    # Save best model
    if avg_d_loss < best_loss:
        best_loss = avg_d_loss
        torch.save(D.state_dict(), "checkpoints/best_D.pth")
        print(f"Best model saved at epoch {epoch+1} with loss {best_loss:.6f}")

# Save last model (optional)
torch.save(D.state_dict(), "checkpoints/D_last.pth")

print("Training complete.")
