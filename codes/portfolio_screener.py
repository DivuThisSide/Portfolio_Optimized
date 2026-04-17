import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error

BEST_PREP = "sg_11_2" 
DATA_ROOT = f"data/processed/{BEST_PREP}"

OPTIMAL_CFG = {
    "lr": 0.001,      
    "hidden": 64,     
    "layers": 1,      
    "batch": 64       
}

SEQ_LENGTH = 30
EPOCHS = 10 

print(" Initializing Live Portfolio Screener...")

class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][1]) 
    return np.array(X), np.array(y)


# SCREENING ENGINE
if __name__ == "__main__":
    screener_results = []

    for file in os.listdir(DATA_ROOT):
        if not file.endswith(".csv"): continue
            
        ticker = file.replace(".csv", "")
        print(f"Analyzing {ticker}...")

        file_path = os.path.join(DATA_ROOT, file)
        df = pd.read_csv(file_path)
        data = df[['Smooth_Close', 'Return', 'MA_10', 'EMA_10']].values

        X, y = create_sequences(data, SEQ_LENGTH)
        if len(X) < 50: continue

        split = int(0.8 * len(X))
        X_train_t = torch.tensor(X[:split], dtype=torch.float32)
        y_train_t = torch.tensor(y[:split], dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X[split:], dtype=torch.float32)
        y_test_t = torch.tensor(y[split:], dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=OPTIMAL_CFG["batch"], shuffle=False)

        model = DynamicGRU(4, OPTIMAL_CFG["hidden"], OPTIMAL_CFG["layers"])
        optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMAL_CFG["lr"])
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(EPOCHS):
            for batch_X, batch_y in train_loader:
                pred = model(batch_X)
                loss = loss_fn(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t).numpy()
            y_test_np = y_test_t.numpy()
            stock_test_mse = mean_squared_error(y_test_np, test_pred)
            
            # Measures what percentage of the time the model correctly guessed the direction (Up/Down)
            directional_accuracy = np.mean((test_pred > 0) == (y_test_np > 0))

        latest_sequence = data[-SEQ_LENGTH:]
        latest_tensor = torch.tensor(latest_sequence, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            predicted_next_return = model(latest_tensor).item()

        # Sum of the actual returns over the last 5 trading days
        recent_trend = np.sum(data[-5:, 1]) 

        screener_results.append({
            "stock": ticker,
            "pred_return": predicted_next_return,
            "mse": stock_test_mse,
            "accuracy": directional_accuracy,
            "trend": recent_trend
        })

    if not screener_results:
        exit()

    df_screen = pd.DataFrame(screener_results)
    df_screen.to_csv("stock_prediction_summary.csv", index=False)
    print("\n Full screening report saved to 'stock_prediction_summary.csv'")