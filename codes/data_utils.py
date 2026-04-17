# Helper functions 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from novel_portfolio_pipeline import INPUT_COLS,MAX_DAILY_RETURN

def get_raw_return(df):
    """Log return from raw Close price. Falls back to Smooth_Close."""
    col = 'Close' if 'Close' in df.columns else 'Smooth_Close'
    price = df[col].replace(0, np.nan)
    lr = np.log(price / price.shift(1))
    return lr.replace([np.inf, -np.inf], np.nan)

def load_and_clean(path):
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['RawReturn'] = get_raw_return(df)
    use = [c for c in INPUT_COLS if c in df.columns] + ['RawReturn']
    df  = df.dropna(subset=use)
    df['RawReturn'] = df['RawReturn'].clip(-MAX_DAILY_RETURN, MAX_DAILY_RETURN)
    if df['RawReturn'].std() < 1e-8:
        return None
    return df

def make_sequences(df, seq_len):
    avail = [c for c in INPUT_COLS if c in df.columns]
    if not avail:
        return None, None, []
    scaler  = MinMaxScaler(feature_range=(-1, 1))
    feat    = scaler.fit_transform(df[avail].values).astype(np.float32)
    raw_ret = df['RawReturn'].values.astype(np.float32)
    X, y = [], []
    for i in range(seq_len, len(feat)):
        X.append(feat[i - seq_len : i])
        y.append(raw_ret[i])
    if not X:
        return None, None, []
    return np.array(X, np.float32), np.array(y, np.float32), avail

def directional_accuracy(y_true, y_pred):
    mask = np.abs(y_true) > 1e-5
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))

def train_one_stock(model, tr, va, epochs, lr, patience, device):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    loss_fn = nn.HuberLoss(delta=0.002)
    best_val, best_wts, no_imp = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            loss_fn(pred.squeeze(), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                vl.append(loss_fn(pred.squeeze(), yb).item())
        v = float(np.mean(vl)) if vl else float('inf')
        sched.step(v)
        if v < best_val:
            best_val = v
            best_wts = {k: w.clone() for k, w in model.state_dict().items()}
            no_imp   = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_wts:
        model.load_state_dict(best_wts)
    return best_val

print("helpers defined ")