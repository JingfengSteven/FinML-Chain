# FinML-Chain: A Blockchain-Integrated Dataset for Enhanced Financial Machine Learning

**🤗 Hugging Face Dataset:** [https://huggingface.co/datasets/StevenJingfeng/FinML](https://huggingface.co/datasets/StevenJingfeng/FinML)

A reproducible, multi-modal dataset integrating **high-frequency Ethereum on-chain transactions** with **low-frequency off-chain social signals** (Discord discussions) to enable proactive Transaction Fee Mechanism (TFM) design under EIP-1559.

## Dataset Overview

The dataset covers two distinct market regimes to isolate volatility effects:

| Period | Date Range | Blocks | Description |
|--------|-----------|--------|-------------|
| **Token Airdrop** | 03/21/2023 – 04/01/2023 | 78,290 | High volatility (ARB airdrop) |
| **Normal** | 06/01/2023 – 07/01/2023 | 213,244 | Stable market conditions |

**Variables:**
- **On-chain**: Timestamp, gas used, gas limit, base fee per gas, gas fraction (α), gas target
- **Off-chain**: Discord chat text from Binance, Uniswap, Ethereum Dev communities
- **Sentiment**: FinBERT-extracted hourly (HS) and daily (DS) sentiment scores

## Benchmark Results

### Baseline Model Comparison

We evaluate Linear Regression, DNN, XGBoost, and LSTM across 5-fold cross-validation.

**Token-Airdrop Period (High Volatility):**

![Token Airdrop Loss](./results/s1.png)
![Token Airdrop Variance](./results/s3.png)

**Key Findings:**
- **DNN achieves superior accuracy** in 7 out of 8 configurations
- Best DNN performance: MSE = 0.0999 at K=5 timesteps (α, β regressors)
- LSTM marginally outperforms DNN only at K=10 (MSE 0.0988 vs 0.0989)
- Linear Regression and XGBoost show significantly higher error (MSE ~0.12)

**Normal Period (Stable Market):**

![Normal Period Loss](./results/s2.png)
![Normal Period Variance](./results/s4.png)

DNNs maintain robust performance with minimal degradation, while other models show significant error increases.

### Improvement with Monotonicity Constraints

To enhance interpretability, we implement Neural Additive Models (NAM) with weak pairwise monotonicity constraints ensuring recent blocks influence predictions more than distant ones.

**Two-Step Training Trajectory (k=3):**

![Two-Step Training Loss](./results/training_loss_2_step.pdf)

Stage 1: Standard backpropagation  
Stage 2: Monotonicity regularization (constraints satisfied without predictive degradation)

**Prediction Comparison:**

Without Monotonicity | With Monotonicity
:------------------:|:------------------:
![Without Mono](./results/No_mono_trained.pdf) | ![With Mono](./results/All_trained.pdf)

Imposing monotonicity preserves the overall prediction pattern while enhancing interpretability and transparency for TFM deployment.

### Improvement with Sentiment Information

We test four configurations combining on-chain (OC) data with daily (DS) and hourly (HS) sentiment:

**Model Performance (MSE) over Two Periods:**

| Configuration | 1 Timestep | 2 Timesteps | 3 Timesteps |
|--------------|------------|-------------|-------------|
| **Period 1: ARB-airdrop** |
| +OC,+DS,+HS | **0.10169** | **0.10056** | **0.10022** ↓ |
| +OC,+DS | 0.10190 | 0.10249 | 0.10150 |
| +OC,+HS | 0.10204 | 0.10213 | 0.10164 |
| +OC only | 0.10290 | 0.10265 | 0.10201 |
| **Period 2: Normal** |
| +OC,+DS,+HS | **0.13593** | **0.13477** | **0.13341** ↓ |
| +OC,+DS | 0.15321 | 0.15381 | 0.15657 |
| +OC,+HS | 0.15459 | 0.15806 | 0.16142 |
| +OC only | 0.18428 | 0.16456 | 0.16089 |

**Visual Comparison:**

![Sentiment 40 datapoints](./results/Combined_prediction_40_data.pdf)
![Sentiment 100 datapoints](./results/Combined_prediction.pdf)

**Key Insights:**
- Sentiment provides **substantial gains during normal periods** (MSE 0.133 vs 0.161 for on-chain-only at 3 timesteps)
- **Marginal improvement during high-activity airdrop** (MSE 0.100 vs 0.102)
- Sentiment partially substitutes for temporal depth: extending timesteps significantly improves on-chain-only settings, but shows limited benefit when sentiment is included

## Repository Structure

```
├── data/
│   ├── ETH-Token-airdrop.csv          # High volatility period (78,290 blocks)
│   ├── ETH-Normal.csv                 # Stable period (213,244 blocks)
│   └── discord_sentiment.json         # FinBERT processed text
├── code/
│   ├── main_dataset_processing.ipynb  # NAM + Monotonicity implementation
│   ├── baseline_models.ipynb          # DNN, XGBoost, LSTM, Linear Regression
│   └── NAM_models.py                  # Neural Additive Model architecture
└── results/
    ├── s1.png, s3.png                 # Token-airdrop comparisons
    ├── s2.png, s4.png                 # Normal period comparisons
    ├── training_loss_2_step.pdf       # Monotonicity training trajectory
    ├── No_mono_trained.pdf            # Predictions without constraints
    ├── All_trained.pdf                # Predictions with constraints
    ├── Combined_prediction_40_data.pdf # Sentiment analysis (40 points)
    └── Combined_prediction.pdf         # Sentiment analysis (100 points)
```

## Quick Start

```python
from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("StevenJingfeng/FinML")

# Access on-chain data
airdrop_data = dataset["token_airdrop"]  # High volatility period
normal_data = dataset["normal_period"]   # Stable period

# Access processed sentiment
sentiment = dataset["discord_sentiment"]  # FinBERT processed
```

## Technical Contributions

1. **Proactive TFM Design**: Dataset enables ML-driven prediction of next-block gas usage for ex-ante fee optimization (vs. reactive EIP-1559)
2. **Interpretable AI**: Monotonicity constraints ensure recent data points influence predictions more than distant ones, satisfying temporal logic without sacrificing accuracy
3. **Multi-modal Integration**: Croissant format seamlessly combines high-frequency blockchain data with low-frequency social signals
4. **Robust Benchmarking**: DNNs identified as optimal architecture across volatile and stable market regimes

## Data Governance

- **Format**: Croissant (MLCommons standard) for interoperability with TensorFlow, PyTorch, Hugging Face
- **Principles**: FAIR (Findable, Accessible, Interoperable, Reusable)
- **Collection**: Google BigQuery (on-chain), DiscordChatExporter (off-chain)
- **Sentiment**: FinBERT financial sentiment analysis

## License

MIT License - see LICENSE file for details.
```
