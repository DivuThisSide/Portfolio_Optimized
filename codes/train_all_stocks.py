import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Importing custom modules and constants
from novel_portfolio_pipeline import (
    DATA_DIR, SEQ_LEN, TRAIN_FRAC, VAL_FRAC, MAX_DAILY_RETURN, 
    CORR_THRESH, OUT_DIR, LAYERS, HIDDEN, DROPOUT, 
    CONV_FILTERS, CONV_KERNEL, BATCH_SIZE, EPOCHS, MC_PASSES, LR, PATIENCE
)
from cnn_bilstm_attention_model import CNNBiLSTMAttention
from stock_gat_model import build_correlation_graph
from data_utils import load_and_clean, make_sequences, train_one_stock, directional_accuracy

# --- FIX: Define the device for CPU execution ---
device = torch.device("cpu")
print(f"Device: {device}")

# 1. Load and Prepare Data
csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
if not csv_files:
    raise RuntimeError(f"No CSV files found in {DATA_DIR}")
print(f"Found {len(csv_files)} CSVs.")

stock_data, raw_ret_dict = {}, {}

for fname in csv_files:
    ticker = fname.replace('.csv', '')
    df = load_and_clean(os.path.join(DATA_DIR, fname))
    
    if df is None or len(df) < SEQ_LEN + 60:
        continue
        
    X, y, avail = make_sequences(df, SEQ_LEN)
    if X is None or len(X) < 50:
        continue
        
    n_tr = int(len(X) * TRAIN_FRAC)
    n_va = int(len(X) * VAL_FRAC)
    
    if len(X) - n_tr - n_va < 10:
        continue
        
    stock_data[ticker] = dict(
        X_tr=X[:n_tr],           y_tr=y[:n_tr],
        X_va=X[n_tr:n_tr+n_va],  y_va=y[n_tr:n_tr+n_va],
        X_te=X[n_tr+n_va:],      y_te=y[n_tr+n_va:],
        n_features=len(avail),
    )
    raw_ret_dict[ticker] = df['RawReturn']

tickers = list(stock_data.keys())
print(f"Usable stocks: {len(tickers)}")
if len(tickers) < 2:
    raise RuntimeError("Need ≥2 usable stocks.")

# 2. Build Correlation Graph
ret_df = pd.DataFrame(raw_ret_dict).dropna()
ret_df.replace([np.inf, -np.inf], np.nan, inplace=True)
ret_df.dropna(axis=1, inplace=True)
ret_df = ret_df.clip(-MAX_DAILY_RETURN, MAX_DAILY_RETURN)

valid = [t for t in tickers if t in ret_df.columns]
ret_df = ret_df[valid]
adj = build_correlation_graph(ret_df, corr_threshold=CORR_THRESH)
gat_order = list(ret_df.columns)

metrics_rows, embeddings = [], {}

def mk_loader(X, y, shuffle):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)

# 3. Training Loop
for i, ticker in enumerate(valid):
    sd = stock_data[ticker]
    model = CNNBiLSTMAttention(
        input_size=sd['n_features'], 
        hidden_size=HIDDEN,
        num_layers=LAYERS,           
        dropout=DROPOUT,
        conv_filters=CONV_FILTERS,   
        conv_kernel=CONV_KERNEL,
    ).to(device)

    print(f"  [{i+1}/{len(valid)}] {ticker}...", end=' ', flush=True)
    t0 = time.time()
    
    # Train
    bv = train_one_stock(
        model,
        mk_loader(sd['X_tr'], sd['y_tr'], True),
        mk_loader(sd['X_va'], sd['y_va'], False),
        EPOCHS, LR, PATIENCE, device
    )
    
    print(f"val={bv:.5f} ({time.time()-t0:.0f}s)", end=' ')
    
    # Ensure directory exists and save
    os.makedirs(f"{OUT_DIR}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{OUT_DIR}/models/{ticker}_novel.pt")

    # 4. MC Inference on TEST SPLIT ONLY
    all_mu, all_sig, all_true, all_pred = [], [], [], []
    for xb, yb in mk_loader(sd['X_te'], sd['y_te'], False):
        xb = xb.to(device)
        
        # Monte Carlo Predictions
        mu, sig = model.mc_predict(xb, MC_PASSES)
        all_mu.extend(mu)
        all_sig.extend(sig)
        all_true.extend(yb.numpy())
        
        # Standard Prediction
        model.eval()
        with torch.no_grad():
            p, _ = model(xb)
        all_pred.extend(p.squeeze().cpu().numpy())

    # Process Metrics
    all_true  = np.array(all_true,  np.float32)
    all_pred  = np.array(all_pred,  np.float32)
    all_mu    = np.clip(np.array(all_mu,  np.float32), -MAX_DAILY_RETURN, MAX_DAILY_RETURN)
    all_sig   = np.clip(np.array(all_sig, np.float32), 1e-6, MAX_DAILY_RETURN)

    mu_hat    = float(np.mean(all_mu))
    sigma_hat = float(np.mean(all_sig))
    rmse      = float(np.sqrt(mean_squared_error(all_true, all_pred)))
    mae       = float(mean_absolute_error(all_true, all_pred))
    d_acc     = directional_accuracy(all_true, all_pred)

    print(f"→ mu={mu_hat:+.5f} σ={sigma_hat:.5f} DirAcc={d_acc:.3f}")
    
    metrics_rows.append(dict(
        ticker=ticker, RMSE=rmse, MAE=mae,
        Dir_Acc=d_acc, mu_hat=mu_hat,
        sigma_hat=sigma_hat
    ))

    # 5. Feature Extraction (Embedding)
    lx = torch.FloatTensor(sd['X_te'][-1:]).to(device)
    with torch.no_grad():
        # Pass through individual components to get context vector
        xc = model.conv(lx.permute(0, 2, 1)).permute(0, 2, 1)
        lo, _ = model.bilstm(xc)
        ctx, _ = model.attention(lo)
        
    embeddings[ticker] = ctx.squeeze(0).cpu()

print("\nTraining complete.")