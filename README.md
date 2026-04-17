# Portfolio_Optimized
This project implements an advanced **AI-based portfolio optimization framework** that combines deep learning for stock prediction with a **risk-aware Mean-CVaR optimization model**.

The system follows a **two-stage pipeline**:
1. Stock Return Prediction using deep learning models  
2. Portfolio Optimization using predicted returns and risk constraints  

---

## 🚀 Features

- Deep Learning Models: MLP, LSTM, GRU, CNN-BiLSTM-Attention  
- Data Denoising using Savitzky-Golay Filter  
- Hyperparameter Optimization using Sparrow Search Algorithm (SSA)  
- Graph Attention Network (GAT) for inter-stock relationships  
- Uncertainty Estimation using Monte Carlo Dropout  
- Risk-aware Portfolio Optimization using Mean-CVaR  
- Market Regime Detection using Hidden Markov Model (HMM)  

---

## 🧠 Methodology

### 🔹 Stage 1: Prediction
- Preprocess stock data (normalization + SG filtering)
- Train models (MLP → LSTM/GRU → CNN-BiLSTM-GAT)
- Generate predicted returns and uncertainty

### 🔹 Stage 2: Optimization
- Select top-performing stocks
- Apply Mean-CVaR optimization
- Adjust weights based on uncertainty and market regime

---

## 📊 Results Summary

- Best prediction RMSE: ~0.01329  
- Directional Accuracy: ~52.4%  
- Maximum Drawdown reduced from −31.2% → −26.1%  
- Improved risk-adjusted performance using uncertainty-aware allocation  

---

## ⚙️ Setup & Installation

1. Clone the repository
```bash
git clone <repo-link>
cd <project-folder>
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Option 1: Full Pipeline (Recommended)
Follow the execution sequence mentioned in:

📄 **order.txt**

This ensures proper pipeline flow:
- Data preprocessing  
- Model training  
- Prediction generation  
- Portfolio optimization  

---

### 🔹 Option 2: Jupyter Notebook (Easy Run)

If the pipeline setup is difficult or fails due to environment issues, use the notebook:

📓 **notebooks/**

- Run all cells step-by-step  
- Useful for quick testing and understanding  
- No strict execution order required  

---

## 📚 Dataset

- Historical stock data (NSE/BSE stocks)
- Features include:
  - OHLC prices  
  - Moving Averages (MA, EMA)  
  - Technical indicators (RSI, MACD)  

---

## 📄 Report

Refer to the **project report PDF** for detailed explanation of:
- Models and architecture  
- Mathematical formulations  
- Experimental analysis  

---

## 👨‍💻 Authors

- Divyansh Bansal  
- Yug Sharma  
- Rachit Jani  
