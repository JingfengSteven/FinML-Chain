<p align="center">
  <a href="https://huggingface.co/datasets/StevenJingfeng/FinML">
    <img src="https://img.shields.io/badge/🤗_Hugging_Face-Dataset-yellow?style=for-the-badge&logo=huggingface" alt="Hugging Face Dataset"/>
  </a>
  <a href="https://github.com/JingfengSteven/FinML-Chain/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"/>
  </a>
  <a href="https://mlcommons.org/working-groups/data/croissant/">
    <img src="https://img.shields.io/badge/Format-Croissant-blue?style=for-the-badge" alt="Croissant Format"/>
  </a>
</p>

<h1 align="center">FinML-Chain</h1>
<p align="center"><b>A Blockchain-Integrated Dataset for Enhanced Financial Machine Learning</b></p>

<p align="center">
  <a href="https://huggingface.co/datasets/StevenJingfeng/FinML"><b>🔗 Dataset on Hugging Face</b></a> •
  <a href="#benchmark-results"><b>📊 Benchmarks</b></a> •
  <a href="#repository-structure"><b>📁 Structure</b></a> •
  <a href="#quick-start"><b>🚀 Quick Start</b></a>
</p>

---

## Overview

FinML-Chain is a reproducible, multi-modal dataset integrating **high-frequency Ethereum on-chain transactions** with **low-frequency off-chain social signals** (Discord discussions) to enable proactive Transaction Fee Mechanism (TFM) design under EIP-1559.

**Key Features:**
- 78,290 blocks during high-volatility ARB token airdrop (03/21–04/01/2023)
- 213,244 blocks during stable market conditions (06/01–07/01/2023)
- FinBERT-processed sentiment from Binance, Uniswap, and Ethereum Dev communities
- Croissant format for seamless ML pipeline integration

---

## Benchmark Results

### Model Performance: Token-Airdrop Period (High Volatility)

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <img src="results/s1.png" width="95%" alt="Token Airdrop Loss"/>
      <br/>
      <sub><b>Average Loss (MSE)</b></sub>
    </td>
    <td width="50%" align="center">
      <img src="results/s3.png" width="95%" alt="Token Airdrop Variance"/>
      <br/>
      <sub><b>Variance Across Folds</b></sub>
    </td>
  </tr>
</table>

**Key Findings:**
- **DNN achieves superior accuracy in 7/8 configurations**
- Best DNN: MSE = 0.0999 at K=5 timesteps (α, β regressors)
- LSTM marginally outperforms only at K=10 (MSE 0.0988 vs 0.0989)

### Model Performance: Normal Period (Stable Market)

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <img src="results/s2.png" width="95%" alt="Normal Period Loss"/>
      <br/>
      <sub><b>Average Loss (MSE)</b></sub>
    </td>
    <td width="50%" align="center">
      <img src="results/s4.png" width="95%" alt="Normal Period Variance"/>
      <br/>
      <sub><b>Variance Across Folds</b></sub>
    </td>
  </tr>
</table>

### Improvement with Monotonicity Constraints

Neural Additive Models (NAM) with weak pairwise monotonicity ensure recent blocks influence predictions more than distant ones.

<p align="center">
  <a href="results/training_loss_2_step.pdf">
    <img src="https://img.shields.io/badge/📉_View_Training_Loss_Trajectory-blue?style=for-the-badge" alt="Training Loss PDF"/>
  </a>
</p>

<p align="center"><sub><b>Two-Step Training (k=3):</b> Stage 1: Standard backpropagation | Stage 2: Monotonicity regularization</sub></p>

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <a href="results/No_mono_trained.pdf">
        <img src="https://img.shields.io/badge/📄_Without_Monotonicity-lightgrey?style=for-the-badge" alt="Without Monotonicity PDF"/>
      </a>
      <br/><br/>
      <sub><b>Without Monotonicity Constraints</b></sub>
    </td>
    <td width="50%" align="center">
      <a href="results/All_trained.pdf">
        <img src="https://img.shields.io/badge/📄_With_Monotonicity-green?style=for-the-badge" alt="With Monotonicity PDF"/>
      </a>
      <br/><br/>
      <sub><b>With Monotonicity Constraints</b></sub>
    </td>
  </tr>
</table>

<p align="center"><i>Click buttons above to view PDF figures. Monotonicity preserves prediction patterns while enhancing interpretability.</i></p>

### Improvement with Sentiment Information

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <a href="results/Combined_prediction_40_data.pdf">
        <img src="https://img.shields.io/badge/📊_40_Datapoints-blue?style=for-the-badge" alt="40 Datapoints PDF"/>
      </a>
      <br/><br/>
      <sub><b>40 Datapoints</b></sub>
    </td>
    <td width="50%" align="center">
      <a href="results/Combined_prediction.pdf">
        <img src="https://img.shields.io/badge/📊_100_Datapoints-blue?style=for-the-badge" alt="100 Datapoints PDF"/>
      </a>
      <br/><br/>
      <sub><b>100 Datapoints</b></sub>
    </td>
  </tr>
</table>

**Sentiment Analysis Results:**

| Configuration | 1 Timestep | 2 Timesteps | 3 Timesteps |
|--------------|------------|-------------|-------------|
| **Period 1: ARB-airdrop** |
| +OC,+DS,+HS | **0.10169** | **0.10056** | **0.10022** ↓ |
| +OC only | 0.10290 | 0.10265 | 0.10201 |
| **Period 2: Normal** |
| +OC,+DS,+HS | **0.13593** | **0.13477** | **0.13341** ↓ |
| +OC only | 0.18428 | 0.16456 | 0.16089 |

**Key Insights:**
- Sentiment provides **substantial gains during normal periods** (MSE 0.133 vs 0.161)
- **Marginal improvement during high-activity airdrop** (MSE 0.100 vs 0.102)
- Sentiment partially substitutes for temporal depth

---

## Repository Structure

```
FinML-Chain/
├── data/
│   ├── ETH-Token-airdrop.csv          # 78,290 blocks (high volatility)
│   ├── ETH-Normal.csv                 # 213,244 blocks (stable)
│   └── discord_sentiment.json         # FinBERT processed text
├── code/
│   ├── main_dataset_processing_code.ipynb  # NAM + Monotonicity
│   ├── baseline_dataset_processing_code.ipynb  # DNN, XGBoost, LSTM, Linear
│   └── NAM_models.py                  # NAM architecture implementation
├── results/
│   ├── s1.png, s2.png, s3.png, s4.png # Benchmark comparisons (PNG)
│   ├── training_loss_2_step.pdf       # Monotonicity training trajectory
│   ├── No_mono_trained.pdf            # Predictions without constraints
│   ├── All_trained.pdf                # Predictions with constraints
│   ├── Combined_prediction_40_data.pdf # Sentiment analysis (40 pts)
│   └── Combined_prediction.pdf        # Sentiment analysis (100 pts)
└── README.md
```

---

## Quick Start

```python
from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("StevenJingfeng/FinML")

# Access on-chain data
airdrop_data = dataset["token_airdrop"]   # 78,290 blocks
normal_data = dataset["normal_period"]    # 213,244 blocks

# Access FinBERT-processed sentiment
sentiment = dataset["discord_sentiment"]
```

---

## Technical Contributions

1. **Proactive TFM Design**: Enables ML-driven next-block gas prediction for ex-ante fee optimization
2. **Interpretable AI**: Monotonicity constraints ensure temporal logic (recent > distant) without accuracy loss
3. **Multi-modal**: Croissant format seamlessly integrates high-frequency blockchain + low-frequency social data
4. **Robust Benchmark**: DNNs identified as optimal across volatile and stable regimes

---

## Data Governance

- **Format**: Croissant (MLCommons standard) for TensorFlow/PyTorch interoperability
- **Principles**: FAIR (Findable, Accessible, Interoperable, Reusable)
- **Sources**: Google BigQuery (on-chain), DiscordChatExporter (off-chain)
- **Sentiment**: FinBERT financial sentiment analysis

---

## License

MIT License - see [LICENSE](https://github.com/JingfengSteven/FinML-Chain/blob/main/LICENSE) file for details.
