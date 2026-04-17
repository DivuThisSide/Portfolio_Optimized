import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

def preprocess_stock(df, config):
    df = df.copy()

    # Format Date column
    df = df.reset_index()
    if 'index' in df.columns:
        df = df.rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Ensure Close column is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.sort_values('Date', inplace=True)
    
    # Handle missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Apply Savitzky-Golay filter if configured
    if config["USE_SG_FILTER"]:
        try:
            df['Smooth_Close'] = savgol_filter(
                df['Close'],
                config["WINDOW_LENGTH"],
                config["POLYORDER"]
            )
        except Exception:
            df['Smooth_Close'] = df['Close']
    else:
        df['Smooth_Close'] = df['Close']

    base_col = 'Smooth_Close' if config["USE_SMOOTHED_FOR_FEATURES"] else 'Close'

    # Calculate daily returns
    df['Return'] = df[base_col].pct_change()

    # Remove outliers using z-score
    z = np.abs((df['Return'] - df['Return'].mean()) / df['Return'].std())
    df = df[z < 3]

    # Calculate moving averages
    df['MA_10'] = df[base_col].rolling(10).mean()
    df['EMA_10'] = df[base_col].ewm(span=10).mean()

    df.dropna(inplace=True)

    # Normalize features
    if config["USE_NORMALIZATION"]:
        features = ['Smooth_Close', 'Return', 'MA_10', 'EMA_10']
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

    return df