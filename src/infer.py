import yaml, torch, pandas as pd
from dataset import get_loader
from model import Discriminator

cfg = yaml.safe_load(open("configs/config.yaml"))
device = torch.device(cfg["device"])

loader, _ = get_loader(cfg["test_csv"], cfg["seq_len"],
                       cfg["label_column"], cfg["batch_size"],
                       False, train_mode=False)

D = Discriminator(loader.dataset.data.shape[1], 64).to(device)
D.load_state_dict(torch.load("checkpoints/D.pth"))
D.eval()

scores, labels = [], []

with torch.no_grad():
    for x,y in loader:
        x = x.to(device)
        s = - D(x)   # anomaly score
        scores.extend(s.cpu().numpy().flatten())
        labels.extend(y.numpy())

pd.DataFrame({"score":scores,"Label":labels}).to_csv(
    "checkpoints/inference_scores.csv", index=False)
