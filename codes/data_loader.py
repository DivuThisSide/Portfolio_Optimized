import os
import yfinance as yf

def download_and_save(tickers, start, end, save_path="data/raw"):
    os.makedirs(save_path, exist_ok=True)

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end)
            
            if not df.empty:
                df.to_csv(os.path.join(save_path, f"{ticker}.csv"))
                
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")