# Research Paper Analysis

## Paper 1: Fin-GAN — Forecasting and Classifying Financial Time Series via GANs

> **Authors:** Milena Vuletić, Felix Prenzel & Mihai Cucuringu (University of Oxford)
> **Published:** Quantitative Finance, Vol. 24, No. 2, pp. 175–199, January 2024

### Core Idea
Fin-GAN uses **Generative Adversarial Networks (GANs)** for **probabilistic forecasting** of financial time series. Unlike traditional point-estimate models (LSTM, ARIMA), Fin-GAN produces full **conditional probability distributions** of future returns, enabling uncertainty quantification.

### Key Contribution — Novel Economics-Driven Loss Function
The main contribution is a custom **generator loss function** that places GANs into a **supervised learning setting** for classification:

```
L_G = J(G) − α·PnL* + β·MSE − γ·SR* + δ·STD
```

| Term | Purpose |
|------|---------|
| **J(G)** | Standard GAN loss (BCE) — learns data distribution |
| **PnL\*** | Smooth approximation to Profit & Loss — correct sign prediction |
| **MSE** | Keep forecasts close to realized values |
| **SR\*** | Maximize Sharpe Ratio — reward risk-adjusted returns |
| **STD** | Minimize PnL variance — reduce strategy risk |

Hyperparameters α, β, γ, δ are determined via a **gradient norm matching** procedure (no manual tuning).

### Architecture — ForGAN
- Built on the **ForGAN** (Koochali et al., 2019) conditional GAN architecture
- Uses **LSTM cells** in both Generator and Discriminator
- **Condition window:** Previous L=10 values → forecasts next value
- **Noise dimension, hidden dimension:** 8 (small due to small datasets)
- **Optimizer:** RMSProp, lr=0.0001
- **Training:** 25 epochs for gradient matching, then 100 epochs for each loss combination
- **Validation:** Choose best loss combination by Sharpe Ratio on validation set

### Data & Experiments
- **Dataset:** Daily stock ETF-excess returns and raw ETF returns from CRSP (Jan 2000 – Dec 2021)
- **Universe:** 22 stocks across 9 sectors + 9 sector ETFs (31 tickers total)
- **Split:** 80% train / 10% validation / 10% test
- **Test period includes COVID-19 pandemic** — making it challenging

### Key Results

| Metric | Fin-GAN | ForGAN (BCE) | LSTM | ARIMA | Long-only |
|--------|---------|-------------|------|-------|-----------|
| Mean Sharpe Ratio | **0.540** | 0.033 | 0.467 | 0.206 | 0.182 |
| Median Sharpe Ratio | **0.413** | -0.092 | 0.214 | 0.204 | 0.194 |
| Portfolio Sharpe Ratio | **2.107** | 0.172 | 2.087 | 0.612 | 0.618 |

- Fin-GAN achieves highest Sharpe Ratios with **lower PnL variance**
- The novel loss function also **alleviates mode collapse** (0% collapse rate vs 67% for ForGAN with He initialization)
- Universality experiments show competitive Sharpe Ratios even on **unseen stocks**

---

## Paper 2: Unsupervised GAN-Based IDS Using Temporal Convolutional Networks and Self-Attention

> **Authors:** Paulo Freitas de Araujo-Filho, Mohamed Naili, Georges Kaddoum, Emmanuel Thepie Fapi, Zhongwen Zhu
> **Published:** IEEE Transactions on Network and Service Management, Vol. 20, No. 4, December 2023

### Core Idea
This paper proposes an **unsupervised Intrusion Detection System (IDS)** that uses a **WGAN** (Wasserstein GAN) to detect **DDoS cyber-attacks** (including zero-day attacks) without requiring labeled attack data.

### Key Innovation — Replacing LSTMs with TCNs and Self-Attention
Most existing GAN-based IDSs use LSTM networks. This paper replaces them with:

| Architecture | Advantages over LSTM |
|---|---|
| **TCNs** (Temporal Convolutional Networks) | Parallel computation, stabler gradients, less memory |
| **Self-Attention** (Multi-Head Attention) | Captures in-depth contextual relationships, constant sequential operations |

### Architecture
- **Framework:** WGAN (Wasserstein GAN)
  - Generator loss: `G_Loss = D(G(z))`
  - Discriminator loss: `D_Loss = D(G(z)) - D(x)`
- **Generator & Discriminator:** Fully connected input/output layers + TCN or self-attention hidden blocks
- **TCN Block:** Dilated causal convolution → ReLU → Normalization → Dropout + Residual connection
- **Self-Attention Block:** Multi-Head Attention → Normalization → Dropout + Residual connection
- **Training:** Only on **benign/normal** network flows; anomalies detected as deviations
- **Deployment:** Edge computing servers for low-latency detection

### Data & Experiments
- **Dataset:** CICDDoS2019 (Canadian Institute for Cybersecurity)
- **Attack types:** Syn, UDP, UDPLag, MSSQL, NetBIOS, LDAP, Portmap
- **Features:** 35 network flow features (e.g., flow duration, packet counts, IAT stats)
- **Training set:** 80% of normal flows from training day
- **Testing set:** 50,000 normal + 50,000 malicious flows from a different day
- **Portmap attack = zero-day** (only in test set)

### Key Results

| Model | AUCROC | Detection Time |
|---|---|---|
| **Our IDS (2 TCN blocks)** | **0.9958** | 3.8× faster than FID-GAN |
| **Our IDS (1 self-attention block)** | **0.9963** | Best accuracy, slightly slower |
| FID-GAN (LSTM-based) | 0.9890 | Slowest |
| ALAD (no temporal modeling) | 0.9340 | Faster but much less accurate |

- **Zero-day detection:** Portmap attack detected with **0.9993 recall**
- Trade-off: More blocks = better accuracy but slower detection
- **At least 3.8× faster** than existing GAN-based IDSs

---

## How Your Project Relates to These Papers

Your codebase in `fin-gan-` implements an **unsupervised GAN-based anomaly detection system** that draws from **Paper 2** (the IDS paper):

| Aspect | Paper 2 | Your Code |
|--------|---------|-----------|
| **Architecture** | WGAN with TCN/Self-Attention | WGAN-GP with GRU (simplified) |
| **Generator** | FC → TCN/SA blocks → FC | GRU → FC |
| **Discriminator** | FC → TCN/SA blocks → FC | GRU → FC (no sigmoid = WGAN critic) |
| **Training** | Normal data only | Benign flows only (`Label == 0`) |
| **Detection** | Discriminator score as anomaly score | Negative discriminator score as anomaly score |
| **Dataset** | CICDDoS2019 | CICDDoS2019 (`merged.csv`) |
| **Evaluation** | ROC-AUC | ROC-AUC (via `scripts/evaluate.py`) |
| **Loss** | Wasserstein loss | Wasserstein loss + Gradient Penalty |
| **Optimizer** | Tuned via Optuna | Adam with β=(0.5, 0.9) |

> [!IMPORTANT]
> Your project is a **simplified implementation** of Paper 2's concept — using GRU instead of TCN/Self-Attention, and WGAN-GP instead of standard WGAN. It does **not** implement the Fin-GAN financial loss function from Paper 1. Paper 1 is included as a related reference for the GAN methodology.

### Key Differences from the Papers
1. **GRU instead of TCN/Self-Attention** — simpler but may not capture long-range dependencies as well
2. **Gradient Penalty** — your code uses WGAN-GP (Gulrajani et al.), which is generally more stable
3. **No hyperparameter optimization** — the papers use Optuna/gradient-norm-matching; your code uses fixed hyperparameters
4. **No encoder network** — unlike FID-GAN, your approach uses the discriminator directly for scoring
