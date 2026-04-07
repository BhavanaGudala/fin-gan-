import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("checkpoints/inference_scores.csv")

# True labels
y_true = df["Label"]

# Discriminator scores
scores = df["score"]

auc = roc_auc_score(y_true, scores)
print("ROC-AUC:", auc)

threshold = scores.quantile(0.95)

# Predicted labels
y_pred = (scores > threshold).astype(int)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, scores)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()