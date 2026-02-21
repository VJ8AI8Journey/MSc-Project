# Finance-Aware AutoRL for Stock Trading
### A Reproducible FinRL Benchmark with Risk-Controlled Tuning and Advanced PPO Variants

This project presents a finance-aware AutoRL framework for stock trading built on top of FinRL.  
It provides a reproducible benchmarking pipeline for deep reinforcement learning (DRL) agents under realistic market constraints, risk-aware objectives, and controlled evaluation splits.

The focus is not just improving returns, but improving **risk-adjusted performance**, robustness, and reproducibility.

---

## 🚀 Project Overview

The framework standardises:

- Data ingestion and feature construction
- Market simulation with transaction costs
- Train / validation / test split (2010–2025)
- Risk-aware evaluation metrics
- Like-for-like agent comparison

All models operate in a shared Gym-compatible trading environment with:

- Explicit transaction costs  
- Sell-before-buy execution logic  
- Reward scaling  
- A global **VIX-based volatility gate** applied uniformly to all agents  

This ensures performance differences arise from algorithmic design, not pipeline inconsistencies.

---

## 🧠 Baseline Algorithms Benchmarked

Using Stable-Baselines3 through FinRL:

- PPO  
- DDPG  
- TD3  
- SAC  

Each algorithm was evaluated under:

1. Default hyperparameters  
2. Meta-gradient in-loop adaptation  
3. Multi-objective Optuna optimisation  

Hyperparameter tuning targeted:

- Sharpe Ratio (maximize)
- Maximum Drawdown (control/minimize)

---

## 🔬 Finance-Aware Enhancements

### 1️⃣ Multi-Objective Hyperparameter Search

Implemented Optuna-based Pareto optimisation to balance return and drawdown rather than optimising a single scalar reward.

---

### 2️⃣ Meta-Gradient Adaptation

Adapted key training parameters online:

- Learning rate  
- PPO clip range  
- Entropy coefficient  

This reduced sensitivity to regime shifts and training instability.

---

### 3️⃣ PPO Variant Family

Eight PPO configurations were evaluated under identical settings:

- PPO (baseline)
- Meta-PPO
- Risk-PPO
- Recurrent PPO (RPPO – LSTM backbone)
- RPPO-Risk
- Transformer-PPO
- Multi-Objective PPO (MOPPO)
- Constraint-PPO

All variants were tested using fixed seeds and identical environment assumptions.

---

## 📊 Key Results (Test Split: 2023–2025)

Best-performing model: **RPPO (LSTM-based PPO)**

- Annual Return: **20.67%**
- Sharpe Ratio: **1.21**
- Maximum Drawdown: **−17.7%**
- Best Calmar ratio among PPO variants

Findings:

- PPO and SAC benefited most from finance-aware tuning.
- TD3 showed higher raw returns but greater instability.
- Temporal memory (LSTM) significantly improved risk-adjusted efficiency.
- Risk shaping reduced volatility and drawdown.
- Transformer-PPO achieved lowest drawdown but lower return.

---

## 📈 Evaluation Protocol

- Train: 2010–2022  
- Validation: 2022–2023  
- Test: 2023–2025  
- Multiple seed repetitions
- Identical cost models across agents
- Risk-adjusted metrics:

  - Annualised Return
  - Annualised Volatility
  - Sharpe Ratio
  - Maximum Drawdown
  - Calmar Ratio

No data leakage (indicators fitted on training only).

---

## 🏗 Framework Components

- Unified FinRL data pipeline
- Gym-compatible StockTradingEnv
- Causal technical indicators (MACD, RSI, Bollinger Bands, SMA)
- Reward scaling & transaction cost modeling
- VIX-based global volatility gate
- Seed-matched repetition for fairness

---

## 🛠 Tech Stack

- Python
- PyTorch
- Stable-Baselines3
- FinRL
- Optuna
- NumPy / Pandas
- Matplotlib

---

## 🎯 Skills Demonstrated

- Deep Reinforcement Learning
- Risk-sensitive policy optimisation
- Multi-objective hyperparameter tuning
- Meta-learning
- Financial time-series modeling
- Reproducible benchmarking design
- LSTM & Transformer integration in RL
- Quantitative evaluation (Sharpe, Drawdown, Calmar)

---

## 🔮 Future Extensions

- CVaR-based optimisation
- Higher-frequency trading environments
- Regime-detection-based adaptation
- Cross-asset portfolio allocation
- More expressive Transformer backbones

---

This project demonstrates how controlled experimental design and finance-aware optimisation improve the reliability and risk efficiency of deep reinforcement learning trading systems.