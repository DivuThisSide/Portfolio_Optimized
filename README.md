# Two-Stage Intelligent Stock Portfolio Optimization using Deep Learning and Mean-CVaR Models
This project implements an advanced **AI-based portfolio optimization framework** that combines deep learning for stock prediction with a **risk-aware Mean-CVaR optimization model**.

The system follows a **two-stage pipeline**:
1. Stock Return Prediction using deep learning models  
2. Portfolio Optimization using predicted returns and risk constraints  

---

## Features

- Deep Learning Models: MLP, LSTM, GRU, CNN-BiLSTM-Attention  
- Data Denoising using Savitzky-Golay Filter  
- Hyperparameter Optimization using Sparrow Search Algorithm (SSA)  
- Graph Attention Network (GAT) for inter-stock relationships  
- Uncertainty Estimation using Monte Carlo Dropout  
- Risk-aware Portfolio Optimization using Mean-CVaR  
- Market Regime Detection using Hidden Markov Model (HMM)  

---

## Methodology

The project is implemented as a **three-phase progressive framework**, where each stage improves upon the previous one.

---

### Phase 1: Baseline Model (MLP)

- A **Multi-Layer Perceptron (MLP)** is used as a non-sequential baseline  
- Input features are preprocessed using:
  - Normalization  
  - Basic feature engineering  
- The model predicts stock returns without capturing temporal dependencies  
- Purpose:
  - Establish a baseline for comparison  
  - Highlight the importance of sequential modeling  

---

### Phase 2: Sequential Modeling + Optimization (LSTM + SSA + Mean-CVaR)

- Apply **Savitzky-Golay (SG) filtering** for noise reduction  
- Train **LSTM/GRU models** to capture temporal patterns in stock data  
- Use **Sparrow Search Algorithm (SSA)** for hyperparameter optimization  
- Generate predicted returns for each stock  
- Perform portfolio allocation using **Mean-CVaR optimization**, focusing on:
  - Maximizing returns  
  - Minimizing tail risk  

---

### Phase 3: Advanced Hybrid Framework  
(CNN-BiLSTM + GAT + Uncertainty-Aware mCVaR)

- Use **1D-CNN + BiLSTM with Attention** for enhanced temporal feature extraction  
- Incorporate **Graph Attention Network (GAT)** to model inter-stock relationships  
- Estimate prediction uncertainty using **Monte Carlo Dropout**  
- Modify portfolio optimization using **Uncertainty-Aware Mean-CVaR**, where:
  - High-uncertainty stocks receive lower allocation  
- Integrate **Hidden Markov Model (HMM)** for market regime detection  

---

## Results Summary

- Best prediction RMSE: ~0.01329  
- Directional Accuracy: ~52.4%  
- Maximum Drawdown reduced from −31.2% → −26.1%  
- Improved risk-adjusted performance using uncertainty-aware allocation  

---

## Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/DivuThisSide/Portfolio_Optimized.git
cd Portfolio_Optimized
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Full Pipeline (Recommended)
Follow the execution sequence mentioned in **order.txt**

This ensures proper pipeline flow:
- Data preprocessing  
- Model training  
- Prediction generation  
- Portfolio optimization  

---

### Option 2: Jupyter Notebook (Easy Run)

If the pipeline setup is difficult or fails due to environment issues, use the **jupyter_notebook**

- Run all cells step-by-step  
- Useful for quick testing and understanding  
- No strict execution order required  

---

## Dataset

- Historical stock data (NSE/BSE stocks)
- Features include:
  - OHLC prices  
  - Moving Averages (MA, EMA)  
  - Technical indicators (RSI, MACD)  

---

## Report

Refer to the **project report PDF** for detailed explanation of:
- Models and architecture  
- Mathematical formulations  
- Experimental analysis  

---

## Authors

- Divyansh Bansal  
- Yug Sharma  
- Rachit Jani  
